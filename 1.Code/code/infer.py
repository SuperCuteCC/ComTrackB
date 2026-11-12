"""CUDA-only inference with PEFT adapters; no NVML/vLLM engine or weight merge."""
import argparse,csv,hashlib,importlib.util,json,math,os,time,traceback,collections
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from answer_contract import validate_answer,requested_unit
ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
S=Path(__file__).resolve().parent;R=S.parent if S.name==a.job else S;J=R/a.job;plan=json.loads((J/'plan-czq.json').read_text());j=next(v for v in plan['jobs'] if v['name']==a.job);ep=j['epochs'];O=J/(f'epoch{ep}-hf-smoke-czq' if a.smoke else f'epoch{ep}-infer32f-czq');O.mkdir(exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES=str(j['gpu']),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',FORCE_QWENVL_VIDEO_READER='decord')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def state(s,**kw):save(O/'status-czq.json',dict(state=s,pid=os.getpid(),updated=time.time(),backend='transformers-peft',**kw))
try:
 import torch,numpy as np
 torch.set_num_threads(2)
 from peft import PeftModel
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from jinja2 import Template
 from decord import VideoReader,cpu
 assert (J/'TRAINING_COMPLETE-czq').exists()
 adapter=J/f'adapter_epoch_{ep}-czq';assert (adapter/'adapter_model.safetensors').exists()
 state('LOADING_MODEL');torch.manual_seed(1);free,total=torch.cuda.mem_get_info();assert free>45*2**30, f'Insufficient free GPU memory: {free/2**30}'
 source=(R/'runtime/evaluation.py');spec=importlib.util.spec_from_file_location('eval_czq',source);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
 tokenizer,processor=e._load_model_tools(Path(j['processor']));fmt=Template(e.FORMAT_PROMPT.read_text())
 model=AutoModelForImageTextToText.from_pretrained(j['base'],trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 model=PeftModel.from_pretrained(model,adapter).eval();model.config.use_cache=True
 template=J/'official_submission_template-czq.csv';records,_,_=e._load_csv(template,J/'assets/videos','.mp4');assert len(records)==3289
 with template.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
 if a.smoke:
  chosen=[]
  for category in ['S2','D2','S3','D3']:
   subset=[i for i,r in enumerate(raw) if r['inference_type'][0]+r['video_type'][1]==category];chosen.extend(subset[:1]+subset[-1:])
  records=[records[i] for i in chosen]
 config=dict(backend='transformers-peft',do_sample=False,max_new_tokens=64,frames=32,max_pixels=262144,seed=1,base=j['base'],adapter=str(adapter),adapter_sha256=sha(adapter/'adapter_model.safetensors'),answer_contract_sha256=sha(R/'answer_contract.py'),whole_answer_validation=True,retry_policy='At most one uniform own-unit six-significant-digit clarification; unresolved remains blank',teacher_sha256=j['teacher_sha256'],template_sha256=sha(template),script_sha256=sha(Path(__file__)))
 assert all(not r.get('answer') for r in records)
 assert all(not row['parsed_value'] for row in raw)
 cf=O/'inference-config.json'
 if cf.exists():assert json.loads(cf.read_text())==config,'Different config; use a new output directory'
 else:save(cf,config)
 done={};journal=O/'generations-hf.jsonl'
 if journal.exists():
  for line in journal.read_text().splitlines():
   x=json.loads(line);assert x['id'] not in done;done[x['id']]=x
 cache=collections.OrderedDict();start=time.time();initial=len(done)
 def video_proc(path):
  if path in cache:cache.move_to_end(path);return cache[path]
  n=len(VideoReader(path,ctx=cpu(0),num_threads=1));v=e.fetch_video(dict(video=path,min_pixels=0,max_pixels=262144,video_fps=2.,nframes=min(32,n//2*2)),return_video_sample_fps=True,return_video_metadata=True)
  cache[path]=v
  if len(cache)>4:cache.popitem(last=False)
  return v
 e._process_video=video_proc
 for record in records:
  rid=record['original_id']
  if rid in done:continue
  prompt=processor.apply_chat_template([dict(role='system',content=e.SYSTEM_PROMPT),dict(role='user',content=e._message_content(record,fmt))],add_generation_prompt=True,tokenize=False,enable_thinking=False)+'<think>\n\n</think>\n\n'
  multimodal,_=e._video_input(record);video,md=multimodal['video'][0];arr=video.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
  inputs=processor(text=prompt,videos=[arr],return_tensors='pt',videos_kwargs={'return_metadata':True,'do_sample_frames':False,'video_metadata':md})
  for k,v in inputs.items():
   if isinstance(v,torch.Tensor):inputs[k]=v.to(device='cuda',dtype=torch.bfloat16 if k=='pixel_values_videos' else v.dtype)
  n=inputs['input_ids'].shape[1];assert n+64<=16384,f'Context too long {rid}: {n}'
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
  response=processor.batch_decode(output[:,n:],skip_special_tokens=True)[0];tokens=output[0,n:].tolist();eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or []);ended=bool(tokens and tokens[-1] in eos);check=validate_answer(response,record['question'],new_tokens=len(tokens),ended_with_eos=ended,require_known_eos=True);value=check['parsed_value'];invalid=not check['accepted']
  attempts=[dict(raw_response=response,parsed_value=value,semantic_invalid=invalid,whole_answer_validation=check,ended_with_eos=ended,generated_token_ids=tokens,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_tokens=n,new_tokens=int(output.shape[1]-n))]
  if invalid:
   clarification='Answer this question in '+requested_unit(record['question'])+'. Return exactly one finite number with at most six significant digits followed by that unit. Do not include explanations.'
   content=e._message_content(record,fmt)+[dict(type='text',text='\n'+clarification)]
   retry=processor.apply_chat_template([dict(role='system',content=e.SYSTEM_PROMPT),dict(role='user',content=content)],add_generation_prompt=True,tokenize=False,enable_thinking=False)+'<think>\n\n</think>\n\n'
   del inputs,output
   inputs=processor(text=retry,videos=[arr],return_tensors='pt',videos_kwargs={'return_metadata':True,'do_sample_frames':False,'video_metadata':md})
   for k,v in inputs.items():
    if isinstance(v,torch.Tensor):inputs[k]=v.to(device='cuda',dtype=torch.bfloat16 if k=='pixel_values_videos' else v.dtype)
   n=inputs['input_ids'].shape[1]
   with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
   response=processor.batch_decode(output[:,n:],skip_special_tokens=True)[0];tokens=output[0,n:].tolist();eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or []);ended=bool(tokens and tokens[-1] in eos);check=validate_answer(response,record['question'],new_tokens=len(tokens),ended_with_eos=ended,require_known_eos=True);value=check['parsed_value'];invalid=not check['accepted']
   attempts.append(dict(raw_response=response,parsed_value=value,semantic_invalid=invalid,whole_answer_validation=check,ended_with_eos=ended,generated_token_ids=tokens,prompt_sha256=hashlib.sha256(retry.encode()).hexdigest(),input_tokens=n,new_tokens=int(output.shape[1]-n)))
  if invalid:value=None
  x=dict(adapter_sha256=config['adapter_sha256'],id=rid,video_id=record['video_id'],raw_response=response,parsed_value=value,parse_failure=invalid,attempts=attempts,input_tokens=n,new_tokens=int(output.shape[1]-n))
  with journal.open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
  done[rid]=x;del inputs,output
  if len(done)%10==0 or a.smoke or len(done)==1:
   elapsed=time.time()-start;rate=elapsed/max(1,len(done)-initial);state('RUNNING',completed=len(done),total=len(records),elapsed_seconds=elapsed,estimated_remaining_seconds=rate*(len(records)-len(done)));print('PROGRESS',len(done),'/',len(records),'seconds_per_question',round(rate,2),flush=True)
 if a.smoke:
  assert len(done)==8;assert all(not x['parse_failure'] for x in done.values()),'Smoke produced unparsed answers'
  state('SMOKE_COMPLETE',completed=8,total=8,elapsed_seconds=time.time()-start);print('SMOKE_COMPLETE',flush=True)
 else:
  assert len(done)==3289 and set(done)=={r['id'] for r in raw}
  save(O/'predictions-FROZEN.json',[done[row['id']] for row in raw]);save(O/'freeze.json',dict(prediction_sha256=sha(O/'predictions-FROZEN.json'),teacher_sha256=j['teacher_sha256'],labels_read_during_inference=False,time=time.time()))
  for row in raw:row['parsed_value']='' if done[row['id']]['parsed_value'] is None else str(done[row['id']]['parsed_value'])
  dest=O/'TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv';tmp=dest.with_suffix('.tmp')
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
  tmp.replace(dest);invalid=[r for r,x in done.items() if x['parse_failure']]
  save(O/'provenance-czq.json',dict(submission_sha256=sha(dest),training_teacher_sha256=j['teacher_sha256'],runtime_inference_config=config,teacher_used_for_training_only=True,hidden_labels_read=False,backend_changed_from_vllm=True,unresolved_output_IDs=invalid,official_score=None))
  state('COMPLETE',completed=3289,total=3289,submission=str(dest),parse_failure_count=len(invalid),elapsed_seconds=time.time()-start)
except Exception:state('FAILED',traceback=traceback.format_exc());raise
