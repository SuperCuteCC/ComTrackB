"""One epoch continued distillation from the immutable measured Track B .863 adapter."""
import argparse,json,math,os,random,time,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig,get_peft_model,PeftModel
from transformers import AutoModelForImageTextToText,AutoProcessor
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--processor',required=True);ap.add_argument('--data',required=True);ap.add_argument('--run',required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--init-adapter');args=ap.parse_args()
run=Path(args.run);run.mkdir(exist_ok=True,parents=True)
def state(s,**kw):
 p=run/'train-status-czq.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'state':s,'pid':os.getpid(),'updated':time.time(),**kw},indent=2));t.replace(p)
try:
 torch.set_num_threads(2);random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
 state('LOADING_MODEL',base=args.base)
 model=AutoModelForImageTextToText.from_pretrained(args.base,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');model.config.use_cache=False
 processor=AutoProcessor.from_pretrained(args.processor,trust_remote_code=True)
 modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']
 model=PeftModel.from_pretrained(model,args.init_adapter,is_trainable=True) if args.init_adapter else get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=modules,task_type='CAUSAL_LM'))
 model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 marker='<|im_start|>';marker_ids=processor.tokenizer.encode(marker,add_special_tokens=False);assert len(marker_ids)==1
 rows=json.loads(Path(args.data).read_text());assert len(rows)>0 and len({r["original_id"] for r in rows})==len(rows)
 def encode(row):
  arr=np.load(row['video']);text=row['prompt_text']+marker+row['answer']+'<|im_end|>'
  inputs=processor(text=text,videos=[arr],return_tensors='pt',videos_kwargs={'return_metadata':True,'do_sample_frames':False,'video_metadata':row['md']})
  ids=inputs['input_ids'][0].tolist();positions=[i for i,v in enumerate(ids) if v==marker_ids[0]];pos=positions[-1]
  for key in ['input_ids','attention_mask','mm_token_type_ids']:
   if key in inputs:inputs[key]=torch.cat((inputs[key][:,:pos],inputs[key][:,pos+1:]),dim=1)
  ids=inputs['input_ids'][0].tolist();labels=[-100]*pos+ids[pos:];assert len(ids)-pos>=2
  assert all(v==-100 for v in labels[:pos]) and labels[pos:]==ids[pos:]
  supervised=processor.tokenizer.decode(ids[pos:],skip_special_tokens=False)
  assert supervised==row['answer']+'<|im_end|>',repr(supervised)
  inputs['labels']=torch.tensor([labels],dtype=torch.long)
  for key,value in inputs.items():
   if isinstance(value,torch.Tensor):inputs[key]=value.to(device='cuda',dtype=torch.bfloat16 if key=='pixel_values_videos' else value.dtype)
  return inputs
 # A real forward/backward from the first shuffled batch validates masks and gradient flow.
 assert model.peft_config['default'].r==16 and model.peft_config['default'].lora_alpha==32
 assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
 params=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.AdamW(params,lr=args.lr,weight_decay=0)
 ga=8;updates=math.ceil(len(rows)/ga)*args.epochs;warmup=max(1,int(updates*.03));step=0;history=[]
 model.train();optim.zero_grad(set_to_none=True)
 for epoch in range(1,args.epochs+1):
  random.shuffle(rows);running=0.;started=time.time()
  for i,row in enumerate(rows):
   inputs=encode(row);loss=model(**inputs).loss
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss at '+row['original_id'])
   group_start=i//ga*ga;divisor=min(ga,len(rows)-group_start);(loss/divisor).backward();running+=float(loss.detach())
   if i==0 and epoch==1:
    nonzero=sum(int(p.grad is not None and torch.count_nonzero(p.grad).item()>0) for p in params)
    assert nonzero>0,'No nonzero LoRA gradients'
    check=dict(id=row['original_id'],answer=row['answer'],loss=float(loss.detach()),nonzero_lora_gradient_tensors=nonzero,answer_only_mask_verified=True,trainable_parameters=sum(p.numel() for p in params),time=time.time())
    (run/'FORWARD_BACKWARD_OK.json').write_text(json.dumps(check,indent=2));print('FORWARD_BACKWARD_OK',json.dumps(check),flush=True)
   if (i+1)%ga==0 or i+1==len(rows):
    step+=1;scale=step/warmup if step<=warmup else max(0.,(updates-step)/(updates-warmup))
    for group in optim.param_groups:group['lr']=args.lr*scale
    gradnorm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optim.step();optim.zero_grad(set_to_none=True)
    if step==1 or step%5==0:
     state('TRAINING',epoch=epoch,step=step,total_steps=updates,rows_seen=i+1,train_loss=running/(i+1),gradient_norm=float(gradnorm),sequence_tokens=int(inputs['input_ids'].shape[1]),elapsed_epoch_seconds=time.time()-started,estimated_remaining_seconds=(time.time()-started)/step*(updates-step))
     print('epoch',epoch,'step',step,'/',updates,'rows',i+1,'loss',running/(i+1),'grad',float(gradnorm),flush=True)
   if ((i+1)%ga==0 or i+1==len(rows)) and (step==20 or step%50==0):
    recovery=dict(step=step,epoch=epoch,next_row=i+1,rows_order=[r['original_id'] for r in rows],running_loss=running,trainable_state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optim.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    tmp=run/'recovery-rolling.tmp';torch.save(recovery,tmp);tmp.replace(run/'recovery-rolling.pt');del recovery
   del inputs,loss
  model.save_pretrained(run/f'adapter_epoch_{epoch}-czq');processor.save_pretrained(run/f'adapter_epoch_{epoch}-czq')
  (run/f'EPOCH_{epoch}_COMPLETE-czq').write_text('ok')
  history.append({'epoch':epoch,'train_loss':running/len(rows),'seconds':time.time()-started})
 shutil.copytree(run/f'adapter_epoch_{args.epochs}-czq',run/'adapter-czq',copy_function=os.link)
 state('SAVING_ADAPTER_ONLY',step=step)
 summary={'initial_adapter':args.init_adapter,'optimizer_reset':True,'base':args.base,'processor':args.processor,'rows':len(rows),'epochs':args.epochs,'lr':args.lr,'lora_r':16,'lora_alpha':32,'grad_accum':ga,'seed':1,'history':history,'training_labels':json.loads((Path(args.data).parent/'manifest-czq.json').read_text())['label_summary'],'merged_weights_retained':False,'teacher_manifest':str(Path(args.data).parent/'manifest-czq.json'),'validation_accuracy':None}
 (run/'summary-czq.json').write_text(json.dumps(summary,indent=2));(run/'TRAINING_COMPLETE-czq').write_text('ok');state('TRAINED',step=step,total_steps=updates)
except Exception:
 state('FAILED',traceback=traceback.format_exc());raise
