#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p63_trajectory_transition_mechanism.py

Purpose:
    Briefly describe the specific computational analysis performed by this
    script and its role in the overall study.

Study context:
    This script is part of the computational workflow used to investigate
    coupling-induced multistability, synchronization loss, re-entrant P/M
    regime selection, and temporal switching in mutually coupled third-order
    phase-locked loops.

Physical regime convention:
    P regime: (D1, D2) = (+, -)
    M regime: (D1, D2) = (-, +)

    P and M denote finite-time detector-phase drift orientations. They should
    not be interpreted automatically as distinct asymptotic attractors.

Inputs:
    - List the required input files, parameters, or outputs from earlier
      scripts.
    - Use "None" when the script is self-contained.

Outputs:
    - List the principal CSV, NPZ, JSON, PNG, PDF, or other generated files.
    - Identify output directories when appropriate.

Manuscript relevance:
    - Section: [Methods/Results section]
    - Figure/Table: [Figure X, Table X, Supplementary Figure SX, or N/A]
    - Principal result: [one-sentence description]

Reproducibility:
    This program is part of the reproducibility repository accompanying the
    manuscript. Numerical classifications and dynamical interpretations should
    be understood within the finite-time framework described in the paper.

Author:
    Hamid Ismail, Ph.D.

Repository:
    Coupled Third-Order PLL Dynamics

License:
    See the LICENSE file in the repository root.
===============================================================================
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

DESIGN=[
 {'transition_id':'e30_r0_b04_t04','ic_id':'e30_r0_b04','edge':30,'x0':[-30.7414393037252,-0.6184411906215073,-0.4627857546586868,2306.5828194495507,0.2087647253360028,-0.703451896101081],'zeta_left':0.89899965,'zeta_right':0.8989996625000001,'left':'P','right':'M'},
 {'transition_id':'e30_r1_b02_t01','ic_id':'e30_r1_b02','edge':30,'x0':[-30.74162510136024,-0.6186889004633828,-0.4622845326162891,2306.5829696949736,0.2091833535274833,-0.7036267328733311],'zeta_left':0.8989998078125,'zeta_right':0.8989998203124999,'left':'P','right':'M'},
 {'transition_id':'e34_r0_b01_t04','ic_id':'e34_r0_b01','edge':34,'x0':[-30.740063019991457,-0.6179420331328805,-0.4640668789878922,2306.583374708984,0.2083947185500864,-0.7039363405934709],'zeta_left':0.89899965,'zeta_right':0.8989996625000001,'left':'P','right':'M'},
 {'transition_id':'e34_r1_b03_t04','ic_id':'e34_r1_b03','edge':34,'x0':[-30.73982331624468,-0.6179415656138276,-0.4641320154590908,2306.5835806248465,0.2085072666269868,-0.704138661019736],'zeta_left':0.8989998453125,'zeta_right':0.8989998578125,'left':'P','right':'M'},
]

def orient(d1,d2,zt=.05):
    if d1 < -zt and d2 > zt:return 'M'
    if d1 > zt and d2 < -zt:return 'P'
    return 'ambiguous'

def circ_R(phi):
    return float(abs(np.mean(np.exp(1j*phi))))

def ac(x,lag):
    x=np.asarray(x,float); n=len(x)
    if n<=lag:return float('nan')
    a=x[:-lag]-np.mean(x[:-lag]); b=x[lag:]-np.mean(x[lag:])
    den=math.sqrt(float(np.dot(a,a)*np.dot(b,b)))
    return float(np.dot(a,b)/den) if den>0 else float('nan')

def slip_intervals(phi,t):
    # crossing times of successive 2pi cells in either direction; descriptive only
    k=np.floor((phi-phi[0])/(2*np.pi)).astype(np.int64)
    idx=np.flatnonzero(np.diff(k)!=0)+1
    if len(idx)<2:return np.array([],float)
    return np.diff(t[idx])

def spectral(x,dt):
    x=np.asarray(x,float)-float(np.mean(x)); n=len(x)
    if n<8:return float('nan'),float('nan')
    p=np.abs(np.fft.rfft(x))**2; f=np.fft.rfftfreq(n,dt)
    p=p[1:]; f=f[1:]
    if not len(p) or float(p.sum())<=0:return float('nan'),float('nan')
    q=p/p.sum(); ent=float(-(q*np.log(q+1e-300)).sum()/np.log(len(q))) if len(q)>1 else 0.0
    return float(f[int(np.argmax(p))]),ent

def metrics(t,Y,z,blocks):
    th1,v1,a1,th2,v2,a2=Y.T
    phi1=th2-th1; phi2=z*th1-th2
    # exact instantaneous detector phase derivatives from state velocities
    m1=v2-v1; m2=z*v1-v2
    dur=float(t[-1]-t[0]); d1=float((phi1[-1]-phi1[0])/dur); d2=float((phi2[-1]-phi2[0])/dur)
    o=orient(d1,d2)
    tv1=float(np.sum(np.abs(np.diff(phi1)))/(2*np.pi*dur)); tv2=float(np.sum(np.abs(np.diff(phi2)))/(2*np.pi*dur))
    si1=slip_intervals(phi1,t); si2=slip_intervals(phi2,t)
    def sicv(si): return float(np.std(si,ddof=1)/np.mean(si)) if len(si)>1 and np.mean(si)>0 else float('nan')
    # block drift variation
    edges=np.linspace(0,len(t)-1,blocks+1,dtype=int); bd1=[];bd2=[]
    for j in range(blocks):
        i0,i1=edges[j],edges[j+1]
        if i1>i0:
            dd=float(t[i1]-t[i0]);bd1.append((phi1[i1]-phi1[i0])/dd);bd2.append((phi2[i1]-phi2[i0])/dd)
    f1,e1=spectral(m1,float(np.median(np.diff(t))));f2,e2=spectral(m2,float(np.median(np.diff(t))))
    return dict(orientation=o,drift1=d1,drift2=d2,
      mismatch1_mean=float(np.mean(m1)),mismatch2_mean=float(np.mean(m2)),
      mismatch1_std=float(np.std(m1)),mismatch2_std=float(np.std(m2)),
      mismatch1_rms=float(np.sqrt(np.mean(m1*m1))),mismatch2_rms=float(np.sqrt(np.mean(m2*m2))),
      mismatch1_ac1=ac(m1,1),mismatch2_ac1=ac(m2,1),mismatch1_ac10=ac(m1,10),mismatch2_ac10=ac(m2,10),
      phi1_R=circ_R(phi1),phi2_R=circ_R(phi2),slip1_rate=tv1,slip2_rate=tv2,
      slip1_interval_mean=float(np.mean(si1)) if len(si1) else float('nan'),slip2_interval_mean=float(np.mean(si2)) if len(si2) else float('nan'),
      slip1_interval_cv=sicv(si1),slip2_interval_cv=sicv(si2),
      block_drift1_std=float(np.std(bd1,ddof=1)),block_drift2_std=float(np.std(bd2,ddof=1)),
      mismatch1_dom_freq=f1,mismatch2_dom_freq=f2,mismatch1_spectral_entropy=e1,mismatch2_spectral_entropy=e2,
      v1_mean=float(np.mean(v1)),v2_mean=float(np.mean(v2)),v1_std=float(np.std(v1)),v2_std=float(np.std(v2)),
      a1_rms=float(np.sqrt(np.mean(a1*a1))),a2_rms=float(np.sqrt(np.mean(a2*a2))))

def run(x0,z,a):
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    x=np.asarray(x0,float)
    if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
    t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs)
    return t,Y,metrics(t,Y,z,a.blocks)

def savej(p,o):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)

def csvout(p,rows,fields):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default='results/author_trajectory_transition_mechanism')
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=6400.)
    ap.add_argument('--dt',type=float,default=.5);ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--blocks',type=int,default=16);ap.add_argument('--trace-stride',type=int,default=20)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.blocks<2 or a.trace_stride<1:ap.error('blocks >=2 and trace-stride >=1 required')
    jobs=[]
    for tr in DESIGN:
        zl,zr=tr['zeta_left'],tr['zeta_right'];w=zr-zl
        pts=[('P_far',zl-2*w),('P_near',zl-w),('left',zl),('q25',zl+.25*w),('mid',zl+.5*w),('q75',zl+.75*w),('right',zr),('M_near',zr+w),('M_far',zr+2*w)]
        for label,z in pts:jobs.append((tr,label,float(z),float((z-zl)/w)))
    if a.diagnostic_only:
        savej(out/'p63_preflight.json',{'status':'PASS','transitions':len(DESIGN),'points_per_transition':9,'planned_integrations':len(jobs),'observe':a.observe,'trace_stride':a.trace_stride,'guardrail':'four representative finite-time P->M crossings; no universal bifurcation claim'})
        print(f'P63 preflight PASS: {len(DESIGN)} transitions x 9 coupling points = {len(jobs)} integrations')
        return
    cf=out/'p63_config.json';cp=out/'p63_checkpoint.json'
    config={'design':DESIGN,'settings':{k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}}
    if a.resume and cf.exists() and json.loads(cf.read_text())!=config:raise ValueError('Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    savej(cf,config); records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];done={r['job_id'] for r in records}
    trace_path=out/'p63_decimated_trajectories.csv';trace_exists=trace_path.exists() and a.resume
    tf=['job_id','transition_id','ic_id','edge','label','zeta','bracket_coordinate','t','phi1','phi2','mismatch1','mismatch2','v1','v2','a1','a2']
    trace_f=open(trace_path,'a' if trace_exists else 'w',newline='');tw=csv.DictWriter(trace_f,fieldnames=tf)
    if not trace_exists:tw.writeheader()
    for tr,label,z,coord in jobs:
        jid=f'{tr["transition_id"]}_{label}'
        if jid in done:continue
        rec={'job_id':jid,'transition_id':tr['transition_id'],'ic_id':tr['ic_id'],'edge':tr['edge'],'label':label,'zeta':z,'bracket_coordinate':coord,'status':'failed'}
        try:
            t,Y,m=run(tr['x0'],z,a);rec.update(status='complete',**m)
            for i in range(0,len(t),a.trace_stride):
                th1,v1,a1,th2,v2,a2=Y[i];tw.writerow({'job_id':jid,'transition_id':tr['transition_id'],'ic_id':tr['ic_id'],'edge':tr['edge'],'label':label,'zeta':z,'bracket_coordinate':coord,'t':float(t[i]),'phi1':float(th2-th1),'phi2':float(z*th1-th2),'mismatch1':float(v2-v1),'mismatch2':float(z*v1-v2),'v1':float(v1),'v2':float(v2),'a1':float(a1),'a2':float(a2)})
            trace_f.flush()
        except Exception as e:rec['error']=repr(e)
        records.append(rec);done.add(jid);savej(cp,records);print(f'{len(records):03d}/{len(jobs)} {jid}: {rec.get("orientation","failed")}',flush=True)
    trace_f.close()
    base=['job_id','transition_id','ic_id','edge','label','zeta','bracket_coordinate','status','orientation','drift1','drift2','mismatch1_mean','mismatch2_mean','mismatch1_std','mismatch2_std','mismatch1_rms','mismatch2_rms','mismatch1_ac1','mismatch2_ac1','mismatch1_ac10','mismatch2_ac10','phi1_R','phi2_R','slip1_rate','slip2_rate','slip1_interval_mean','slip2_interval_mean','slip1_interval_cv','slip2_interval_cv','block_drift1_std','block_drift2_std','mismatch1_dom_freq','mismatch2_dom_freq','mismatch1_spectral_entropy','mismatch2_spectral_entropy','v1_mean','v2_mean','v1_std','v2_std','a1_rms','a2_rms','error']
    csvout(out/'p63_metrics.csv',records,base)
    # transition summaries: orientations across 9 points and endpoint jumps
    sums=[]
    for tr in DESIGN:
        rr=[r for r in records if r['transition_id']==tr['transition_id'] and r.get('status')=='complete'];rr=sorted(rr,key=lambda r:r['bracket_coordinate'])
        left=next((r for r in rr if r['label']=='left'),None);right=next((r for r in rr if r['label']=='right'),None)
        s={'transition_id':tr['transition_id'],'ic_id':tr['ic_id'],'edge':tr['edge'],'n_complete':len(rr),'orientation_sequence':'|'.join(f'{r["label"]}:{r["orientation"]}' for r in rr)}
        if left and right:
            for k in ['drift1','drift2','mismatch1_rms','mismatch2_rms','mismatch1_std','mismatch2_std','phi1_R','phi2_R','slip1_rate','slip2_rate','slip1_interval_mean','slip2_interval_mean','mismatch1_spectral_entropy','mismatch2_spectral_entropy']:
                s['delta_'+k]=right[k]-left[k]
        sums.append(s)
    sfields=sorted({k for s in sums for k in s},key=lambda x:(x not in ['transition_id','ic_id','edge','n_complete','orientation_sequence'],x))
    csvout(out/'p63_transition_summary.csv',sums,sfields)
    complete=sum(r.get('status')=='complete' for r in records);failed=len(records)-complete
    summary={'n_planned':len(jobs),'n_records':len(records),'n_complete':complete,'n_failed':failed,'n_transitions':len(DESIGN),'points_per_transition':9,'orientation_counts':{o:sum(r.get('orientation')==o for r in records) for o in ['P','M','ambiguous']},'transition_summaries':sums}
    savej(out/'p63_summary.json',summary)
    savej(out/'p63_report.json',{'scope':'P63 trajectory-level dynamical mechanism across four representative persistence-verified P60 P->M crossings','settings':config['settings'],'summary':summary,'indicator_notes':{'slip_rate':'total variation of unwrapped detector phase/(2*pi*observation duration)','slip_interval':'intervals between changes of 2pi phase cell; descriptive for running/slipping trajectories','spectral_entropy':'normalized Shannon entropy of one-sided mismatch periodogram excluding DC','bracket_coordinate':'(zeta-zeta_left)/(zeta_right-zeta_left); left=0, right=1'},'interpretation_guardrails':['The four crossings are representative finite-time regime-selection transitions, not a complete parameter-space census.','Nearby sample points were chosen within the locally scanned P60 neighborhood; unresolved narrower re-entrant windows can still exist between them.','Abrupt changes in trajectory statistics across a P/M bracket do not by themselves establish a dynamical bifurcation.','Autocorrelation and spectral changes in a deterministic smooth ODE are descriptive and are not, alone, evidence of critical slowing down.','All classifications and trajectory statistics are finite-time; long observations improve reproducibility but do not prove asymptotic attraction.']})
    print(f'P63 finished: {complete}/{len(records)} complete; {failed} failed')
if __name__=='__main__':main()

