#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p62_local_basin_resilience.py

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

def classify(t,Y,z):
    p1=Y[:,3]-Y[:,0];p2=z*Y[:,0]-Y[:,3];dur=float(t[-1]-t[0])
    d1=float((p1[-1]-p1[0])/dur);d2=float((p2[-1]-p2[0])/dur)
    return orient(d1,d2),d1,d2

def run(x0,z,transient,observe,dt,rtol,atol,max_abs):
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    x=np.asarray(x0,float)
    if transient:x=integrate(x,p,transient,dt,rtol,atol,max_abs)[1][-1]
    t,Y=integrate(x,p,observe,dt,rtol,atol,max_abs)
    return classify(t,Y,z)

def savej(p,o):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)

def csvout(p,rows,fields):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default='results/author_local_basin_resilience')
    ap.add_argument('--radii',default='2e-7,5e-7,1e-6,2e-6,5e-6,1e-5,2e-5')
    ap.add_argument('--directions',type=int,default=3)
    ap.add_argument('--refine-levels',type=int,default=8)
    ap.add_argument('--transient',type=float,default=400.)
    ap.add_argument('--observe',type=float,default=3200.)
    ap.add_argument('--verify-observe',type=float,default=6400.)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13)
    ap.add_argument('--max-abs',type=float,default=1e7);ap.add_argument('--seed',type=int,default=6101)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();radii=[float(v) for v in a.radii.split(',')]
    if not radii or min(radii)<=0 or any(radii[i]>=radii[i+1] for i in range(len(radii)-1)):ap.error('radii must be positive strictly increasing')
    if a.directions<1 or a.refine_levels<1:ap.error('directions/refine-levels must be >=1')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cp=out/'p62_checkpoint.json';cf=out/'p62_config.json'
    config={'design':DESIGN,'settings':{k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}}
    if a.resume and cf.exists() and json.loads(cf.read_text())!=config:raise ValueError('Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    rng=np.random.default_rng(a.seed);base_dirs=[]
    for _ in range(a.directions):
        v=rng.standard_normal(6);v/=np.linalg.norm(v);base_dirs.append(v)
    rays=[]
    for di,v in enumerate(base_dirs):
        for sign in (-1,1):rays.append((di,sign,sign*v))
    anchors=[]
    for tr in DESIGN:
        for side in ('left','right'):
            anchors.append({'anchor_id':f'{tr["transition_id"]}_{side}','transition_id':tr['transition_id'],'ic_id':tr['ic_id'],'edge':tr['edge'],'side':side,'expected':tr[side],'zeta':tr['zeta_'+side],'x0':tr['x0']})
    primary_n=len(anchors)*(1+len(rays)*len(radii))
    max_refine=len(anchors)*len(rays)*a.refine_levels
    max_verify=len(anchors)*len(rays)*2
    if a.diagnostic_only:
        savej(out/'p62_preflight.json',{'status':'PASS','anchors':len(anchors),'rays_per_anchor':len(rays),'radii':radii,'primary_integrations':primary_n,'maximum_refinement_integrations':max_refine,'maximum_long_verifications':max_verify,'maximum_total':primary_n+max_refine+max_verify,'definition':'first detected switch along sampled ray; no monotonic/global basin-distance claim'})
        print(f'P62 preflight PASS: {len(anchors)} anchors x {len(rays)} rays x {len(radii)} radii + 8 baselines = {primary_n} primary; max total {primary_n+max_refine+max_verify}')
        return
    savej(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];done={r['job_id'] for r in records}
    def execute(job_id,anchor,x0,observe,phase,radius=None,di=None,sign=None,level=None):
        nonlocal records,done
        if job_id in done:return next(r for r in records if r['job_id']==job_id)
        rec={'job_id':job_id,'phase':phase,'anchor_id':anchor['anchor_id'],'transition_id':anchor['transition_id'],'ic_id':anchor['ic_id'],'edge':anchor['edge'],'side':anchor['side'],'expected':anchor['expected'],'zeta':anchor['zeta'],'radius':radius,'direction':di,'sign':sign,'level':level,'observe':observe,'status':'failed'}
        try:
            o,d1,d2=run(x0,anchor['zeta'],a.transient,observe,a.dt,a.rtol,a.atol,a.max_abs)
            rec.update(status='complete',orientation=o,drift1=d1,drift2=d2,reproduced=(o==anchor['expected']))
        except Exception as e:rec['error']=repr(e)
        records.append(rec);done.add(job_id);savej(cp,records);print(f'{len(records):04d} {job_id}: {rec.get("orientation","failed")}',flush=True);return rec
    # exact baselines and radial scans
    for an in anchors:
        execute(an['anchor_id']+'_baseline',an,an['x0'],a.observe,'baseline',radius=0.)
        xbase=np.asarray(an['x0'],float)
        for di,sgn,v in rays:
            for ri,r in enumerate(radii):
                execute(f'{an["anchor_id"]}_ray_d{di}_{sgn:+d}_r{ri}',an,xbase+r*v,a.observe,'scan',radius=r,di=di,sign=sgn)
    # identify first sampled expected->nonexpected switch on each ray, then refine bracket
    brackets=[]
    for an in anchors:
      xbase=np.asarray(an['x0'],float)
      for di,sgn,v in rays:
        seq=[]
        b=next(r for r in records if r['job_id']==an['anchor_id']+'_baseline');seq.append((0.,b))
        for ri,r in enumerate(radii):seq.append((r,next(q for q in records if q['job_id']==f'{an["anchor_id"]}_ray_d{di}_{sgn:+d}_r{ri}')))
        first=None
        for k in range(1,len(seq)):
            prev_r,prev=seq[k-1];cur_r,cur=seq[k]
            if prev.get('orientation')==an['expected'] and cur.get('orientation') in ('P','M') and cur.get('orientation')!=an['expected']:
                first=[prev_r,cur_r,prev['orientation'],cur['orientation']];break
        if first is None:continue
        lo,hi,olo,ohi=first
        # bisection only within observed opposite-orientation endpoint bracket
        for lev in range(a.refine_levels):
            mid=(lo+hi)/2
            rr=execute(f'{an["anchor_id"]}_ray_d{di}_{sgn:+d}_bis{lev}',an,xbase+mid*v,a.observe,'refine',radius=mid,di=di,sign=sgn,level=lev)
            if rr.get('orientation')==an['expected']:lo=mid;olo=rr['orientation']
            elif rr.get('orientation') in ('P','M'):hi=mid;ohi=rr['orientation']
            else:break
        # verify final endpoints at long horizon
        vl=execute(f'{an["anchor_id"]}_ray_d{di}_{sgn:+d}_verify_lo',an,xbase+lo*v,a.verify_observe,'verify',radius=lo,di=di,sign=sgn)
        vh=execute(f'{an["anchor_id"]}_ray_d{di}_{sgn:+d}_verify_hi',an,xbase+hi*v,a.verify_observe,'verify',radius=hi,di=di,sign=sgn)
        brackets.append({'anchor_id':an['anchor_id'],'transition_id':an['transition_id'],'edge':an['edge'],'side':an['side'],'expected':an['expected'],'direction':di,'sign':sgn,'scan_lo':first[0],'scan_hi':first[1],'refined_lo':lo,'refined_hi':hi,'midpoint':(lo+hi)/2,'width':hi-lo,'lo_orientation_3200':olo,'hi_orientation_3200':ohi,'lo_orientation_6400':vl.get('orientation'),'hi_orientation_6400':vh.get('orientation'),'persistent':vl.get('orientation')==an['expected'] and vh.get('orientation') in ('P','M') and vh.get('orientation')!=an['expected']})
    fields=['job_id','phase','anchor_id','transition_id','ic_id','edge','side','expected','zeta','radius','direction','sign','level','observe','status','orientation','drift1','drift2','reproduced','error']
    csvout(out/'p62_integrations.csv',records,fields)
    bfields=['anchor_id','transition_id','edge','side','expected','direction','sign','scan_lo','scan_hi','refined_lo','refined_hi','midpoint','width','lo_orientation_3200','hi_orientation_3200','lo_orientation_6400','hi_orientation_6400','persistent']
    csvout(out/'p62_first_switch_brackets.csv',brackets,bfields)
    # anchor summary and global summary
    arows=[]
    for an in anchors:
        bb=[b for b in brackets if b['anchor_id']==an['anchor_id']];pers=[b for b in bb if b['persistent']]
        mids=[b['midpoint'] for b in pers]
        arows.append({'anchor_id':an['anchor_id'],'edge':an['edge'],'side':an['side'],'expected':an['expected'],'n_rays':len(rays),'n_first_switch_detected':len(bb),'n_persistent':len(pers),'persistent_fraction':len(pers)/len(rays),'min_persistent_midpoint':min(mids) if mids else None,'median_persistent_midpoint':float(np.median(mids)) if mids else None,'max_persistent_midpoint':max(mids) if mids else None})
    csvout(out/'p62_anchor_resilience.csv',arows,['anchor_id','edge','side','expected','n_rays','n_first_switch_detected','n_persistent','persistent_fraction','min_persistent_midpoint','median_persistent_midpoint','max_persistent_midpoint'])
    complete=sum(r['status']=='complete' for r in records);failed=len(records)-complete
    sm={'n_records':len(records),'n_complete':complete,'n_failed':failed,'n_anchors':len(anchors),'n_rays_per_anchor':len(rays),'n_primary_planned':primary_n,'n_first_switch_brackets':len(brackets),'n_persistent_first_switch_brackets':sum(b['persistent'] for b in brackets),'anchor_resilience':arows}
    savej(out/'p62_summary.json',sm)
    savej(out/'p62_report.json',{'scope':'P62 first-detected local perturbation switching radii around eight representative P61 anchors','settings':config['settings'],'summary':sm,'interpretation_guardrails':['Because P60/P61 show interleaved and non-monotonic regime selection, a first-detected switching radius is not the mathematical distance to a global/asymptotic basin boundary.','The radial scan can miss switching windows narrower than or between the prescribed radius samples.','Bisection refines only an already observed expected/opposite orientation bracket; it does not assume behavior beyond that bracket is monotone.','All classifications are finite-time. A persistent 6400-unit endpoint check strengthens numerical reproducibility but does not prove asymptotic attraction.','The four representative P60 transitions/eight anchors do not establish global resilience throughout state or parameter space.']})
    print(f'P62 finished: {complete}/{len(records)} complete; {len(brackets)} first-switch brackets; {sum(b["persistent"] for b in brackets)} persistent at 6400')
if __name__=='__main__':main()

