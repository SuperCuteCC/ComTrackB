from pathlib import Path
import json,hashlib,csv,math,os,sys,importlib.util,shutil,subprocess,time
R=Path(__file__).resolve().parent;plan=json.loads((R/'PLAN_FROZEN.json').read_text());J=R/'student'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
assert sha(R/'teacher-FROZEN.csv')==plan['teacher']['sha256']
for x in plan['initial_adapter']['files']:assert sha(R/'init-adapter'/Path(x['path'].replace('\\','/')).name)==x['sha256']
base=Path(plan['base']['remote_path']);assert sha(base/'model.safetensors')==plan['base']['historical_weight_sha256']
source=Path(plan['training_cache_source'])
old=json.loads(source.read_text());norm=[{k:r[k] for k in ['original_id','video_id','md','prompt_text']} for r in old];ih=hashlib.sha256(json.dumps(norm,sort_keys=True).encode()).hexdigest();assert ih==plan['cache_contract']['input_hash']
for r in old:r['video']=str(Path(plan['training_cache_directory'])/Path(r['video']).name)
byid={r['original_id']:r for r in old};assert len(byid)==len(old)==3289
for sample in plan['cache_contract']['frame_samples']:assert sha(byid[sample['id']]['video'])==sample['sha256']
cache={r['video'] for r in old};assert len(cache)==568 and all(Path(p).is_file() for p in cache)
teacher=list(csv.DictReader((R/'teacher-FROZEN.csv').open(encoding='utf-8-sig',newline='')));template=list(csv.DictReader((R/'input-template-NO-ANSWERS.csv').open(encoding='utf-8-sig',newline='')));units=json.loads((R/'teacher-units.json').read_text())
from answer_contract import requested_unit
assert all(units[r['id']]==requested_unit(r['question']) for r in teacher)
assert len(teacher)==len(template)==3289 and all(not r['parsed_value'] for r in template)
assert all(set(a)==set(b) and all(a[k]==b[k] for k in a if k!='parsed_value') for a,b in zip(teacher,template))
rawvideos=Path('/home/ubuntu/data/quantiphy_czq/QuantiPhy');assert len(list(rawvideos.glob('*.mp4')))==568
spec=importlib.util.spec_from_file_location('helper',R/'runtime/evaluation.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
from jinja2 import Template
tokenizer,processor=e._load_model_tools(base);fmt=Template(e.FORMAT_PROMPT.read_text());records,_,_=e._load_csv(R/'input-template-NO-ANSWERS.csv',rawvideos,'.mp4')
data=[]
for t,blank,record in zip(teacher,template,records):
 sid=t['id'];src=byid[sid];assert sid==record['original_id'] and src['video_id']==t['video_id'] and src['category']==record['category']
 assert math.isfinite(float(t['parsed_value'])) and sid in units and units[sid]
 prompt=processor.apply_chat_template([dict(role='system',content=e.SYSTEM_PROMPT),dict(role='user',content=e._message_content(record,fmt))],add_generation_prompt=True,tokenize=False,enable_thinking=False)+'<think>\n\n</think>\n\n'
 assert prompt==src['prompt_text'],'Prompt mismatch '+sid
 row={k:src[k] for k in ['original_id','video_id','video','md','prompt_text','category']}
 row.update(answer=t['parsed_value']+' '+units[sid],label_kind='MAIN7493_UNCHANGED_FROZEN_PSEUDO_LABEL_NOT_GROUND_TRUTH');data.append(row)
assert len(data)==3289
J.mkdir(exist_ok=False);(J/'assets').mkdir();(J/'assets/videos').symlink_to(rawvideos,target_is_directory=True)
shutil.copyfile(R/'input-template-NO-ANSWERS.csv',J/'official_submission_template-czq.csv');shutil.copyfile(R/'infer.py',J/'infer.py')
save(J/'train.json',data);save(J/'plan-czq.json',dict(jobs=plan['jobs']));save(J/'manifest-czq.json',dict(label_summary='Exact original parsed_value strings from latest MAIN7493 teacher plus own-question units; all 3289 rows; inherited provenance gaps preserved; not ground truth',teacher_sha256=plan['teacher']['sha256'],input_metadata_hash=ih,training_data_sha256=sha(J/'train.json'),old_answers_discarded=True))
save(R/'PREFLIGHT_OK.json',dict(time=time.time(),rows=3289,videos=568,base_sha256=sha(base/'model.safetensors'),cache_samples_verified=True,all_3289_prompts_reconstructed_and_matched=True,all_3289_teacher_metadata_matched=True,unit_counts={u:list(units.values()).count(u) for u in sorted(set(units.values()))}))
import torch,transformers,peft,numpy,decord,platform
save(R/'SOFTWARE.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,peft=peft.__version__,numpy=numpy.__version__,decord=decord.__version__))
print('PREFLIGHT_OK',flush=True)
