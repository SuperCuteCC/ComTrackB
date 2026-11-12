"""Exactly one frozen continuation and one resumable pure-student inference."""
from pathlib import Path
import fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
R=Path(__file__).resolve().parent
J=R/'student'
plan=json.loads((R/'PLAN_FROZEN.json').read_text())
py=sys.executable
lock=(R/'worker.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def state(value,**kw):
    p=R/'STATUS.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value,pid=os.getpid(),updated=time.time(),**kw),indent=2))
    tmp.replace(p)

def resource_check(stage):
    assert shutil.disk_usage(R).free>2*2**30,'Less than 2 GiB free disk; no task/data deletion authorized'
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    devices={int(line.split(',')[0]):line.split(',')[1].strip() for line in result.stdout.splitlines()}
    gpu=plan['jobs'][0]['gpu'];uuid=devices[gpu]
    owners=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    assert not any(line.split(',')[0].strip()==uuid for line in owners.splitlines()),'Selected GPU has a compute owner; no other process will be touched'
    code="import torch,json;f,t=torch.cuda.mem_get_info();assert f>80*2**30,(f/2**30);a=torch.randn(256,256,device='cuda',dtype=torch.bfloat16);b=a@a;torch.cuda.synchronize();assert torch.isfinite(b).all();print(json.dumps(dict(free_GiB=f/2**30,total_GiB=t/2**30,bf16_test=True,device=torch.cuda.get_device_name())))"
    check=subprocess.run([py,'-B','-c',code],capture_output=True,text=True,timeout=60)
    assert check.returncode==0,check.stdout+'\n'+check.stderr
    (R/('RESOURCE_BEFORE_'+stage+'.json')).write_text(json.dumps(dict(time=time.time(),port=50019,gpu=gpu,uuid=uuid,cuda=json.loads(check.stdout),disk_free_GiB=shutil.disk_usage(R).free/2**30,selected_gpu_compute_owners=[]),indent=2))

def stage(name,args):
    assert not (R/'STOP_REQUESTED').exists(),'STOP_REQUESTED'
    with (R/(name+'.log')).open('a') as output:
        child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,cwd=R)
        state(name,child_pid=child.pid,log=str(R/(name+'.log')))
        rc=child.wait()
    assert rc==0,name+' failed exit '+str(rc)

try:
    if not (J/'TRAINING_COMPLETE-czq').exists():
        assert not (J/'train-status-czq.json').exists(),'Existing training state; inspect rolling recovery rather than retraining'
        assert (R/'CACHE_TRANSFER_VERIFIED.json').exists(),'Training cache must be verified first'
        resource_check('TRAIN')
        if not (R/'PREFLIGHT_OK.json').exists():stage('PREPARING',[py,'-B','-u',str(R/'prepare.py')])
        resource_check('TRAIN')
        j=plan['jobs'][0]
        assert j['lr']==2e-5 and j['epochs']==1 and plan['initial_adapter']['checkpoint_official_score']==.863
        stage('TRAINING',[py,'-B','-u',str(R/'train.py'),'--base',j['base'],'--processor',j['processor'],'--data',str(J/'train.json'),'--run',str(J),'--lr',str(j['lr']),'--epochs','1','--init-adapter',j['init_adapter']])
    adapter=J/'adapter_epoch_1-czq'
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir() if p.is_file()}
    (R/'ADAPTER_FROZEN.json').write_text(json.dumps(dict(time=time.time(),files=files),indent=2))
    if not (R/'INFERENCE_COMPLETE').exists():
        resource_check('INFERENCE')
        stage('INFERENCE',[py,'-B','-u',str(J/'infer.py'),'--job','student'])
        (R/'INFERENCE_COMPLETE').write_text('ok')
    stage('VERIFYING',[py,'-B','-u',str(R/'finalize.py')])
    state('COMPLETE',official_score=None)
except Exception:
    state('FAILED',traceback=traceback.format_exc())
    raise
