from pathlib import Path
import csv,json,hashlib,math,time,shutil
R=Path(__file__).resolve().parent;J=R/'student';O=J/'epoch1-infer32f-czq'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
name='TRACKB-FROM0863-MAIN7493-LR2E5-E1-UNTESTED.csv';p=O/name
template=read(J/'official_submission_template-czq.csv');rows=read(p);gens=[json.loads(s) for s in (O/'generations-hf.jsonl').read_text().splitlines()];byid={g['id']:g for g in gens};ah=sha(J/'adapter_epoch_1-czq/adapter_model.safetensors')
assert len(rows)==len(template)==len(gens)==len(byid)==3289
invalid=[];unit_mismatches=[]
for row,t in zip(rows,template):
 assert len(row)==10 and row['id']==t['id'] and all(row[k]==t[k] for k in t if k!='parsed_value')
 g=byid[row['id']];assert g['adapter_sha256']==ah and g['video_id']==row['video_id'];v=g['parsed_value'];assert row['parsed_value']==('' if v is None else str(v))
 assert 1<=len(g['attempts'])<=2 and g['raw_response']==g['attempts'][-1]['raw_response']
 if v is None:invalid.append(row['id'])
 else:
  assert math.isfinite(v)
  if 'speed' in row['question'].lower():assert v>=0
config=json.loads((O/'inference-config.json').read_text());assert config['adapter_sha256']==ah
save('VALIDATION.json',dict(rows=3289,unique_ids=3289,columns=10,metadata_and_order_preserved=True,all_outputs_checkpoint_verified=True,no_teacher_fallback=True,unresolved_ids=invalid,official_score=None))
# Freeze predictions and raw generations before any teacher comparison.
save('FREEZE.json',dict(time=time.time(),csv_sha256=sha(p),generations_sha256=sha(O/'generations-hf.jsonl'),predictions_sha256=sha(O/'predictions-FROZEN.json'),adapter_sha256=ah,official_score=None))
teacher=read(R/'teacher-FROZEN.csv');assert [t['id'] for t in teacher]==[r['id'] for r in rows]
fit=[];exact=0
for t,row in zip(teacher,rows):
 if not row['parsed_value']:fit.append(0);continue
 a=float(row['parsed_value']);b=float(t['parsed_value']);exact+=a==b
 fit.append(sum(abs(a-b)/max(abs(b),1e-9)<1-x for x in [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])/10 if b!=0 else 0)
save('TEACHER_FIT_NOT_OFFICIAL_SCORE.json',dict(mean_teacher_fit=sum(fit)/len(fit),exact_matches=exact,not_official_accuracy=True,not_evidence_target_reached=True))
plan=json.loads((R/'PLAN_FROZEN.json').read_text());save('PROVENANCE.json',dict(base=plan['base'],initial_adapter=plan['initial_adapter'],teacher=plan['teacher'],teacher_record=plan['teacher_record'],training=plan['training'],inference_config=config,software=json.loads((R/'SOFTWARE.json').read_text()),inherited_source_eligibility_gaps_preserved=True,closed_outputs_used_during_development=True,inference_public_weights_only=True,official_score=None,upload_performed=False))
save('HANDOFF_CODEX1.json',dict(state='COMPLETE_UNTESTED',csv=str(p),adapter=str(J/'adapter_epoch_1-czq'),csv_sha256=sha(p),adapter_sha256=ah,official_score=None,automatic_upload=False,second_round=False,required_next_action='Receive candidate and obtain actual official result; retain actual score. No further training authorized.'))
(R/'REPORT.txt').write_text('Completed one 2e-5 continuation epoch from measured .863 with the unchanged MAIN7493 pseudo-label teacher and one pure-student 3289-question inference. Official score untested. Invalid outputs: '+str(invalid)+'\nTeacher fit is not official accuracy. Source eligibility gaps inherited. No official upload or second round.\n')
print('COMPLETE_UNTESTED',sha(p),flush=True)
