'use client';
import { useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { apiClient } from '@/utils/api';
import { actionError } from '@/utils/jobs';
import { nextPair, type Review, type Row, type Vote } from './review';

const defaults = { model: 'OpenMOSS-Team/MOSS-Image-SFT', teacher_steps: 50, steps: 20, copies: 2,
  width: 512, height: 512, seed: 42, cfg: 3, shift: 5, sampler: 'euler', schedule: 'moss', negative: '' };
type Recipe = typeof defaults & { prompts: string[]; sigmas?: number[]; teacher_sigmas?: number[] };
type Ready = { recipe: Recipe; rows: Row[]; teacher_job: string; fingerprint: string };
const empty = (): Review => ({ votes: [], excluded: [], skipped: [] });
const inputClass = 'block rounded bg-gray-800 px-3 py-2 my-1 w-full';
const btn = 'rounded bg-gray-700 hover:bg-gray-600 px-3 py-2 disabled:opacity-40';

function Preview({job, item, loaded}: {job: string; item: string; loaded:(id:string)=>void}) {
  const [src, setSrc] = useState('');
  const [failed,setFailed] = useState(false);
  useEffect(() => {
    let alive = true, url = '';
    setSrc(''); setFailed(false);
    apiClient.get('/api/distill', { params: {id: job, item}, responseType: 'blob' }).then(r => {
      if (alive) { url = URL.createObjectURL(r.data); setSrc(url); }
    }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [job,item]);
  return src ? <img src={src} alt="Anonymous teacher candidate" onLoad={()=>loaded(item)} className="w-full object-contain max-h-[42vh]" /> :
    <p>{failed ? 'Preview unavailable' : 'Loading preview…'}</p>;
}

export default function DistillPage() {
  const router = useRouter();
  const query = useSearchParams();
  const [name,N] = useState('moss-speed');
  const [gpu,G] = useState('0');
  const [r,R] = useState(defaults);
  const [prompts,P] = useState('');
  const [sigmas,S] = useState('');
  const [teacherSigmas,TS] = useState('');
  const [doReview,Q] = useState(false);
  const [updates,U] = useState(1000), [rank,K] = useState(8), [lr,L] = useState(.0001);
  const [weight,W] = useState(.1), [useDpo,D] = useState(false);
  const [jobs,J] = useState<{id:string;name:string}[]>([]);
  const [source,F] = useState(query.get('teacher') || ''), [ready,A] = useState<Ready | null>(null);
  const [review,V] = useState<Review>(empty), [group,H] = useState(0);
  const [shown,show] = useState<Record<string,boolean>>({});
  const imageLoaded = useCallback((id:string)=>show(s=>s[id]?s:{...s,[id]:true}),[]);
  const [busy,B] = useState(false), [error,E] = useState('');
  const storage = (id:string, key:string) => 'moss-review:'+id+':'+key;
  async function refresh() {
    try {
      const res = await apiClient.get('/api/jobs', {params:{job_type:'moss_distill'}});
      J(res.data.jobs.filter((j:{job_config:string; status:string}) => {
        const cfg = typeof j.job_config === 'string' ? JSON.parse(j.job_config) : j.job_config;
        return j.status === 'completed' && cfg?.config.process[0]?.stage === 'prepare';
      }));
    } catch (e) { E(actionError(e)); }
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    let alive = true;
    A(null); V(empty()); H(0); D(false); E(''); show({});
    if (source) apiClient.get('/api/distill',{params:{id:source}}).then(res => {
      if (!alive) return;
      const data:Ready = res.data;
      const saved = localStorage.getItem(storage(source,data.fingerprint));
      const old = saved ? JSON.parse(saved) : empty();
      V(old); A(data); N(data.teacher_job+'-student');
    }).catch(e => {if(alive) E(actionError(e));});
    return () => {alive=false;};
  },[source]);
  function change(v: Review) {
    if (!ready) return;
    V(v);
    try { localStorage.setItem(storage(source,ready.fingerprint), JSON.stringify(v)); }
    catch { E('Browser storage is full; these votes are only in memory until you save the student job.'); }
  }
  const pair = ready && nextPair(ready.rows, group, review);
  const mark = (id:string) => change({...review, excluded: review.excluded.includes(id)
    ? review.excluded.filter(k => k!==id) : [...review.excluded,id]});
  function vote(result:Vote[2] | 'skip') {
    if (!pair) return;
    change(result === 'skip' ? {...review, skipped:[...review.skipped,pair]} :
      {...review, votes:[...review.votes,[pair[0],pair[1],result]]});
  }
  async function save() {
    E(''); B(true);
    try {
      if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(name)) throw Error('Enter a job name without spaces or path separators.');
      if (!/^\d+$/.test(gpu)) throw Error('Choose one GPU index.');
      if (source && !ready) throw Error('Wait for the teacher cache to load.');
      const recipe:Recipe = ready?.recipe ?? {...r, prompts:prompts.split('\n').map(s=>s.trim()).filter(Boolean)};
      if (!recipe.prompts.length || new Set(recipe.prompts).size !== recipe.prompts.length) throw Error('Enter distinct prompts.');
      if (!ready && r.schedule === 'explicit') {
        recipe.sigmas = sigmas.trim().split(/[\s,]+/).map(Number);
        const g=recipe.sigmas;
        if(g.length!==r.steps+1 || g[0]!==1 || g.at(-1)!==0 || g.some((s,i)=>!Number.isFinite(s)||(i>0 && s>=g[i-1]))) throw Error('Enter exactly X+1 descending sigmas, including 1 and 0.');
      }
      if (!ready && teacherSigmas.trim()) {
        const g=teacherSigmas.trim().split(/[\s,]+/).map(Number);
        if(g.length!==r.teacher_steps+1 || g[0]!==1 || g.at(-1)!==0 || g.some((s,i)=>!Number.isFinite(s)||(i>0 && s>=g[i-1]))) throw Error('Teacher sigmas must run from 1 to 0 and match the teacher step count.');
        recipe.teacher_sigmas=g;
      }
      if (recipe.teacher_steps<=recipe.steps || recipe.width%16 || recipe.height%16) throw Error('Teacher steps must exceed student steps; dimensions must be multiples of 16.');
      if (!ready && doReview && r.copies<2) throw Error('Pair review requires at least two candidates per prompt.');
      if (ready && review.excluded.length===ready.rows.length) throw Error('All regression targets are excluded.');
      if (ready && useDpo && !review.votes.some(v=>v[2]!=='tie')) throw Error('DPO requires a decisive comparison.');
      const stage = ready ? 'student' : doReview ? 'prepare' : 'all';
      const proc = {type:'moss_step_distill',device:'cuda:0',stage,recipe,
        student:{updates,rank,lr,dpo_weight:ready && useDpo ? weight : 0},
        ...(ready ? {teacher_job:ready.teacher_job,fingerprint:ready.fingerprint,votes:review.votes,excluded:review.excluded} : {})};
      const res = await apiClient.post('/api/jobs',{name,gpu_ids:gpu,job_type:'moss_distill',
        job_config:{job:'extension',config:{name,process:[proc]}}});
      router.push('/jobs/'+res.data.id);
    } catch(e) { E(e instanceof Error && !('response' in e) ? e.message : actionError(e)); }
    finally { B(false); }
  }
  return <div className="max-w-4xl mx-auto space-y-5 p-6">
    <h1 className="text-2xl font-semibold">MOSS step distillation</h1>
    <p>Train a CFG-1 speed LoRA from a frozen MOSS-Image SFT teacher. Use a merged personalized checkpoint to preserve its look.</p>
    <label className="block">Workflow<select className={inputClass} value={source} onChange={e=>F(e.target.value)}>
      <option value="">New teacher → student</option>
      {jobs.map(j=><option key={j.id} value={j.id}>Review teacher: {j.name}</option>)}
    </select></label>
    <button className={btn} onClick={refresh}>Refresh finished teachers</button>
    <form className="space-y-4" onSubmit={e=>{e.preventDefault();void save();}}>
    {!source && <>
      <label className="block">Teacher checkpoint (hub ID or local BF16 path)<input className={inputClass} required value={r.model} onChange={e=>R({...r,model:e.target.value})}/></label>
      <label className="block">Training prompts · one per line<textarea className={inputClass} rows={5} required value={prompts} onChange={e=>P(e.target.value)}/></label>
      <label className="block">Or load a UTF-8 prompt list <input type="file" accept=".txt,text/plain"
        onChange={async e=>{const file=e.target.files?.[0];if(file)try{P(await file.text());}catch{E('Could not read the prompt list.');}}}/></label>
      <label className="block">Teacher negative prompt<input className={inputClass} value={r.negative} onChange={e=>R({...r,negative:e.target.value})}/></label>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {([['teacher_steps','Teacher Euler steps'],['steps','Student steps'],['copies','Seeds per prompt'],['seed','First seed'],
          ['width','Width'],['height','Height'],['cfg','Teacher CFG'],['shift','Flow shift']] as const).map(([k,label])=>
        <label key={k}>{label}<input className={inputClass} type="number" required min={k==='seed'?0:k==='shift'?0.01:1}
          step={k==='shift'||k==='cfg'?'any':1} value={r[k]} onChange={e=>R({...r,[k]:Number(e.target.value)})}/></label>)}</div>
      <label className="block">Student sampler<select className={inputClass} value={r.sampler} onChange={e=>R({...r,sampler:e.target.value})}>
        <option value="euler">Euler · one call per step</option><option value="heun">Heun · two calls except the final step</option></select></label>
      <label className="block">Student sigma schedule<select className={inputClass} value={r.schedule} onChange={e=>R({...r,schedule:e.target.value})}>
        <option value="moss">Native MOSS</option><option value="flow_uniform">Shifted uniform flow time</option><option value="explicit">Exact custom sigmas</option></select></label>
      {r.schedule==='explicit' && <textarea className={inputClass} required placeholder="1, …, 0" value={sigmas} onChange={e=>S(e.target.value)}/>}
      <details><summary>Override the native teacher sigma schedule</summary><label>Exact teacher sigmas (including terminal zero)
        <textarea className={inputClass} placeholder="Leave empty for native MOSS; paste exact sigmas to reproduce another Euler workflow" value={teacherSigmas} onChange={e=>TS(e.target.value)}/></label></details>
      <p>Teacher trajectory cache: about {(prompts.split('\n').filter(s=>s.trim()).length*r.copies*(r.steps+1)*r.width*r.height*2/2**30).toFixed(2)} GiB, plus embeddings and optional PNGs.</p>
      <label><input type="checkbox" checked={doReview} onChange={e=>Q(e.target.checked)}/> Save teacher previews and finish this job for blind review before training</label>
    </>}
    {ready && <section className="space-y-3 border border-gray-700 rounded p-4">
      <label>Prompt group<select className={inputClass} value={group} onChange={e=>H(Number(e.target.value))}>{ready.recipe.prompts.map((p,i)=><option key={i} value={i}>{i+1}. {p}</option>)}</select></label>
      <p>{ready.recipe.prompts[group]}</p>
      {pair ? <><div className="grid grid-cols-2 gap-2">{pair.map((id,i)=><div key={id}><Preview job={source} item={id} loaded={imageLoaded}/>
        <button type="button" disabled={!shown[id]} className={btn} onClick={()=>mark(id)}>{review.excluded.includes(id)?'Restore':'Exclude'} {i?'B':'A'} {review.excluded.includes(id)?'(excluded)':'from regression'}</button></div>)}</div>
        <div className="flex flex-wrap gap-2">{(['a','b','tie','skip'] as const).map((v,i)=><button key={v} type="button" className={btn} disabled={v!=='skip' && !pair.every(id=>shown[id])} onClick={()=>vote(v)}>{['Prefer A','Prefer B','Tie','Skip'][i]}</button>)}</div>
      </> : <p>All pairs in this group reviewed. Choose another group or save the student job.</p>}
      <p>{review.votes.length} votes · {review.excluded.length} excluded targets. Votes autosave in this browser; saving the job copies them into its config.</p>
      <button type="button" className={btn} disabled={!review.votes.length} onClick={()=>change({...review,votes:review.votes.slice(0,-1)})}>Undo last vote</button>
      <button type="button" className={btn} disabled={!review.skipped.length} onClick={()=>change({...review,skipped:[]})}>Revisit skipped pairs</button>
      <label className="block"><input type="checkbox" checked={useDpo} onChange={e=>D(e.target.checked)}/> Add DPO to trajectory loss using same-prompt Elo extremes</label>
      {useDpo && <label>DPO loss weight<input className={inputClass} type="number" required min=".000001" step="any" value={weight} onChange={e=>W(+e.target.value)}/></label>}
    </section>}
    <div className="grid sm:grid-cols-3 gap-3">
      <label>Job name<input className={inputClass} required value={name} onChange={e=>N(e.target.value)}/></label>
      <label>GPU index<input className={inputClass} required value={gpu} onChange={e=>G(e.target.value)}/></label>
      {(!doReview || source) && <><label>Student updates<input type="number" className={inputClass} required min={1} value={updates} onChange={e=>U(+e.target.value)}/></label>
      <label>LoRA rank<input type="number" className={inputClass} required min={1} value={rank} onChange={e=>K(+e.target.value)}/></label>
      <label>Learning rate<input type="number" className={inputClass} required min=".000000001" step="any" value={lr} onChange={e=>L(+e.target.value)}/></label></>}
    </div>
    {error && <p role="alert" className="text-red-300">{error}</p>}
    <button disabled={busy || (!!source && !ready)} className={btn} type="submit">{busy?'Saving…':ready?'Save reviewed student job':doReview?'Save teacher generation job':'Save generation + training job'}</button>
    <p className="text-sm text-gray-400">Saving does not start the GPU. Start the job from its overview when ready.</p>
    </form>
  </div>;
}
