"""Two-parameter, all-luminosity ZTF comparison; imports the unchanged engine.

Run ``.venv/Scripts/python.exe -B analysis/simplified_luminosity_function/ztf.py``.
The selection is deliberately identical to analysis.lf_rate_medians: its
phase-at-peak rise/fade cuts are not replaced by a different physical model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import lf_rate_medians as previous  # noqa: E402
from grb_detect.constants import DAY_S  # noqa: E402
from grb_detect.params import GPC_TO_CM  # noqa: E402

L_REF = 1e32
OUTPUT = Path(__file__).resolve().parent / "results"


@dataclass
class Case:
    mode_name: str
    window: str
    mode: dict
    model: object
    night_factor: float
    F_lim_Jy: float
    L_dec: float
    L_fid: float
    conversion_1day_hz: float
    sky_fraction: float
    t_exp_s: float


@lru_cache(maxsize=4)
def build_case(mode_name, window):
    mode, model, night = previous.build_state(None, mode_name, window)
    exposure = float(model.t_exp_s(mode["N_exp"], mode["t_cad_s"]))
    flux = float(model.F_lim_Jy(np.asarray(exposure)))
    factor = 4 * math.pi * model.phys.D_euc_cm**2 * 1e-23
    fid = factor * model.derived.F_dec_Jy
    return Case(mode_name, window, mode, model, night, flux, factor*flux,
                fid, model.L0_erg_s()/fid,
                float(model.f_Omega(mode["N_exp"])), exposure)


def _shape(case, t):
    """Existing engine's continuous two-segment on-axis decline."""
    m = case.model
    td, tj = m.derived.t_dec_s, m.derived.t_j_s
    a = -m.pls.alpha_II_temporal(m.phys.p)
    b = -m.pls.alpha_III_temporal(m.phys.p)
    t = np.maximum(np.asarray(t, dtype=float), td)
    return np.where(t<tj, (t/td)**(-a), (tj/td)**(-a)*(t/tj)**(-b))


def _profile(case, q):
    m, s = case.model, case.mode
    q = np.asarray(q, dtype=float)
    tp, tf, k, ise = m._rise_fade_windows(
        q, s["i_det"], np.asarray(s["t_cad_s"]),
        previous.SELECTION["s_fade"], previous.SELECTION["s_rise"], "discrete")
    c = s["t_cad_s"]
    wf = np.clip(tf-tp, 0, c)
    gamma = 2/k
    fp = _shape(case, tp)
    req = float(m._t_req_s(s["i_det"], np.asarray(c)))
    gw = np.minimum(fp, _shape(case, tp+req))
    rise = fp*ise**2
    uc = np.clip(tp*np.expm1(np.log(rise/gw)/gamma), 0, wf)
    return tp, wf, gamma, gw, rise, uc


def filtered_moment(case, q, alpha):
    """H_b(q)=cadence^-1 integral_0^wf min(g_window,g_rise(u))^b du."""
    beta = -float(alpha)-1
    if not 0 < beta < 1.5:
        raise ValueError("Detected all-L rate requires -2.5 < alpha < -1")
    tp, wf, gamma, gw, rise, uc = _profile(case, q)
    z = 1-gamma*beta
    log_ratio = np.log1p((wf-uc)/(tp+uc))
    # Stable across the logarithmic antiderivative gamma*beta=1.
    f = np.ones_like(z)
    np.divide(np.expm1(z*log_ratio), z, out=f, where=np.abs(z)>1e-11)
    f = np.where(np.abs(z)>1e-11, f, log_ratio)
    ramp = rise**beta*tp*(1+uc/tp)**z*f
    return (uc*gw**beta+ramp)/case.mode["t_cad_s"]


def _q_from_time(case, t):
    td, tj = case.model.derived.t_dec_s, case.model.derived.t_j_s
    if t <= td:
        return None
    return 1+(t/tj)**(.375 if t<tj else .5)


def _breaks(case):
    m, s = case.model, case.mode
    qn = float(m.derived.q_nr)
    cuts = [0., 1., float(m.derived.q_dec), 1.5, 2., qn]
    req = float(m._t_req_s(s["i_det"], np.asarray(s["t_cad_s"])))
    times = [m.derived.t_j_s-req]
    tf = m._t_first_caps(s["i_det"], np.asarray(s["t_cad_s"]), .3, "discrete")
    for t in tf:
        times.extend([float(t), float(t)-s["t_cad_s"]])
    for t in times:
        q = _q_from_time(case, t)
        if q is not None and 0<q<qn:
            cuts.append(q)
    # Include both boundaries of the analytic u antiderivative's active pieces.
    initial = sorted(set(cuts))
    def transitions(q):
        tp,wf,gamma,gw,rise,uc = _profile(case, np.asarray(q))
        raw = tp*np.expm1(np.log(rise/gw)/gamma)
        return float(raw), float(raw-wf)
    for left,right in zip(initial[:-1],initial[1:]):
        grid=np.linspace(left+1e-11,right-1e-11,33)
        for which in (0,1):
            vals=[transitions(q)[which] for q in grid]
            for a,b,fa,fb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
                if fa*fb<0:
                    for _ in range(48):
                        mid=(a+b)/2; fm=transitions(mid)[which]
                        if fa*fm<=0: b=mid
                        else: a=mid;fa=fm
                    cuts.append((a+b)/2)
    return sorted(set(cuts))


@lru_cache(maxsize=64)
def quadrature(mode_name, window, nq=96):
    case=build_case(mode_name,window)
    x,w=leggauss(nq)
    cuts=_breaks(case)
    q=np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(cuts[:-1],cuts[1:])])
    weights=np.concatenate([(b-a)*w/2 for a,b in zip(cuts[:-1],cuts[1:])])
    return q,weights


def rate_per_amplitude(case, alpha, nq=96):
    beta=-alpha-1
    q,w=quadrature(case.mode_name,case.window,nq)
    angular=float(np.dot(w*q,filtered_moment(case,q,alpha)))
    d=case.model.phys.D_euc_cm/GPC_TO_CM
    coefficient=(4*math.pi*d**3*case.model.phys.theta_j_rad**2
                 *case.sky_fraction*case.night_factor
                 /(beta*(3-2*beta))*(case.L_dec/L_REF)**(-beta))
    return coefficient*angular


def distance_quantile(case,alpha,quantile):
    return case.model.phys.D_euc_cm/GPC_TO_CM*quantile**(1/(2*alpha+5))


def _weighted_quantile(x,mass,p):
    # Midpoint placement avoids the one-node bias of end-bin interpolation.
    c=(np.cumsum(mass)-mass/2)/mass.sum()
    return float(np.interp(p,c,x))


def mixture(case,alpha,nq=96,nu=32):
    """Independent q,u quadrature of threshold components, used for L CDFs."""
    q,wq=quadrature(case.mode_name,case.window,nq)
    tp,wf,gamma,gw,rise,uc=_profile(case,q)
    x,wu=leggauss(nu)
    all_h=[];all_w=[];all_q=[]
    for lo,hi in ((np.zeros_like(uc),uc),(uc,wf)):
        u=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*x/2
        h=np.minimum(gw[:,None],rise[:,None]*(1+u/tp[:,None])**(-gamma[:,None]))
        weight=wq[:,None]*q[:,None]*(hi-lo)[:,None]*wu/2/case.mode["t_cad_s"]
        all_h.append(h.ravel());all_w.append(weight.ravel())
        all_q.append(np.broadcast_to(q[:,None],h.shape).ravel())
    h=np.concatenate(all_h);weight=np.concatenate(all_w);qq=np.concatenate(all_q)
    keep=weight>0
    beta=-alpha-1
    return case.L_dec/h[keep],weight[keep]*h[keep]**beta,qq[keep]


def conditional_luminosity_cdf(luminosity,threshold,beta):
    z=np.asarray(luminosity)/threshold
    return np.where(z<=1,(2*beta/3)*z**(1.5-beta),
                    1-(1-2*beta/3)*z**(-beta))


def mixture_cdf(luminosity,threshold,mass,beta):
    return float(np.dot(mass,conditional_luminosity_cdf(luminosity,threshold,beta))/mass.sum())


def _bisect_log_cdf(cdf,target,low=-300.,high=300.):
    for _ in range(75):
        mid=(low+high)/2
        if cdf(10.**mid)<target: low=mid
        else: high=mid
    return 10**((low+high)/2)


def statistics(case,alpha,A,nq=96,nu=32):
    rate=rate_per_amplitude(case,alpha,nq)*A
    q,w=quadrature(case.mode_name,case.window,nq)
    mass=q*w*filtered_moment(case,q,alpha)
    qm=_weighted_quantile(q,mass,.5)
    threshold,mixmass,_=mixture(case,alpha,nq,nu)
    beta=-alpha-1
    med=_bisect_log_cdf(lambda lum: mixture_cdf(lum,threshold,mixmass,beta),.5,
                      math.log10(threshold.min())-20,math.log10(threshold.max())+320)
    return dict(raw_rate=rate,rate=rate*previous.COVERAGE,q_med=qm,
                theta_med_rad=qm*case.model.phys.theta_j_rad,
                theta_med_deg=math.degrees(qm*case.model.phys.theta_j_rad),
                frac_q_lt_1=float(mass[q<1].sum()/mass.sum()),
                frac_q_lt_1p5=float(mass[q<1.5].sum()/mass.sum()),
                frac_q_gt_2=float(mass[q>2].sum()/mass.sum()),
                D_med_Gpc=distance_quantile(case,alpha,.5),
                D_90_Gpc=distance_quantile(case,alpha,.9),
                L_med_spectral=med,L_med_1day_erg_s=med*case.conversion_1day_hz,
                angular_moment=float(mass.sum()),
                independent_moment_relative_error=float(abs(mixmass.sum()/mass.sum()-1)),
                t_exp_s=case.t_exp_s,F_lim_Jy=case.F_lim_Jy)


def optimal_amplitude(coefficients,windows=None,coverage=previous.COVERAGE):
    logs=[math.log(coverage*k/previous.TARGETS[mode]["rate"])
          for (window,mode),k in coefficients.items() if windows is None or window in windows]
    return math.exp(-.5*(max(logs)+min(logs)))


def evaluate(alpha,A=None,nq=96,distributions=False,nu=32,optimize_windows=None):
    co={(window,mode):rate_per_amplitude(build_case(mode,window),alpha,nq)
        for window in previous.WINDOWS for mode in previous.MODE_KEYS}
    if A is None:A=optimal_amplitude(co,optimize_windows)
    result=dict(alpha=float(alpha),A=float(A),windows={})
    for window in previous.WINDOWS:
        modes={}
        for mode in previous.MODE_KEYS:
            case=build_case(mode,window)
            if distributions:stats=statistics(case,alpha,A,nq,nu)
            else:stats=dict(raw_rate=co[window,mode]*A,rate=co[window,mode]*A*previous.COVERAGE,
                            D_med_Gpc=distance_quantile(case,alpha,.5))
            modes[mode]=stats
        score=previous.score(modes)
        result["windows"][window]=dict(modes=modes,score=score,passes=score<=1)
    selected=[w["score"] for key,w in result["windows"].items()
              if optimize_windows is None or key in optimize_windows]
    result["score"]=max(selected)
    result["passes"]=result["score"]<=1
    result["classification"]=previous.classify(result)
    return result


def finite_statistics(case,alpha,A,log_lo,log_hi,nq=192,nu=48):
    """Exact cutoff conditioning of the all-L mixture; no LF renormalization."""
    threshold,mass,q=mixture(case,alpha,nq,nu)
    beta=-alpha-1;lo=10**log_lo;hi=10**log_hi
    flo=conditional_luminosity_cdf(lo,threshold,beta)
    fhi=conditional_luminosity_cdf(hi,threshold,beta)
    selected=mass*(fhi-flo);frac=float(selected.sum()/mass.sum())
    order=np.argsort(q);qs=q[order];ms=selected[order]
    uq,starts=np.unique(qs,return_index=True);qm=np.add.reduceat(ms,starts)
    def dcdf(d):
        if d<=0:return 0.
        return d**(3-2*beta)*float(np.dot(mass,
            conditional_luminosity_cdf(hi/d**2,threshold,beta)-
            conditional_luminosity_cdf(lo/d**2,threshold,beta))/selected.sum())
    def dquantile(probability):
        low,high=0.,1.
        for _ in range(60):
            mid=(low+high)/2
            if dcdf(mid)<probability:low=mid
            else:high=mid
        return (low+high)/2*case.model.phys.D_euc_cm/GPC_TO_CM
    low_cdf=float(np.dot(mass,flo)/mass.sum())
    lum_med=_bisect_log_cdf(lambda lum: mixture_cdf(lum,threshold,mass,beta),
                            low_cdf+.5*frac,log_lo,log_hi)
    return dict(rate=rate_per_amplitude(case,alpha,nq)*A*previous.COVERAGE*frac,
                D_med_Gpc=dquantile(.5),D_90_Gpc=dquantile(.9),
                q_med=_weighted_quantile(uq,qm,.5),retained_fraction=frac,
                L_med_spectral=lum_med,
                frac_q_lt_1=float(selected[q<1].sum()/selected.sum()),
                frac_q_lt_1p5=float(selected[q<1.5].sum()/selected.sum()),
                frac_q_gt_2=float(selected[q>2].sum()/selected.sum()),
                omitted_faint=float(np.dot(mass,flo)/mass.sum()),
                omitted_bright=float(np.dot(mass,1-fhi)/mass.sum()))


def verify_engine(alpha,A,nq=2000,nd=1000,cutoffs=((26.,40.),(24.,42.))):
    checks=[]
    for window in previous.WINDOWS:
        for mode in previous.MODE_KEYS:
            case=build_case(mode,window)
            conversion=math.log10(case.conversion_1day_hz)
            for loglo,loghi in cutoffs:
                analytic=finite_statistics(case,alpha,A,loglo,loghi)
                engine=previous.distribution_at((alpha,loglo+conversion,loghi+conversion),mode,window,nq,nd)
                beta=-alpha-1
                rho=A/beta*((10**loglo/L_REF)**(-beta)-(10**loghi/L_REF)**(-beta))
                engine_rate=engine["rate"]*rho/case.model.phys.rho_grb_gpc3_yr
                lum_med=analytic['L_med_spectral']
                med_rate=previous.rate_at((alpha,loglo+conversion,math.log10(lum_med)+conversion),
                                         mode,window,nq)['rate']
                rho_med=A/beta*((10**loglo/L_REF)**(-beta)-(lum_med/L_REF)**(-beta))
                engine_lum_cdf=med_rate/engine['rate']*rho_med/rho
                errors=dict(rate=abs(engine_rate/analytic["rate"]-1),
                            D_median=abs(engine["D_med_Gpc"]/analytic["D_med_Gpc"]-1),
                            D_90=abs(engine["D_90_Gpc"]/analytic["D_90_Gpc"]-1),
                            q_median=abs(engine["q_med"]-analytic["q_med"]),
                            luminosity_median_cdf=abs(engine_lum_cdf-.5),
                            **{k:abs(engine[k]-analytic[k]) for k in
                               ('frac_q_lt_1','frac_q_lt_1p5','frac_q_gt_2')})
                checks.append(dict(alpha=alpha,A=A,window=window,mode=mode,
                                   spectral_log10_bounds=[loglo,loghi],analytic=analytic,
                                   engine_rate=engine_rate,engine_D_med_Gpc=engine["D_med_Gpc"],
                                   engine_q_med=engine["q_med"],engine_luminosity_median_cdf=engine_lum_cdf,
                                   errors=errors,
                                   passes=all(v<.01 for v in errors.values()),N_q=nq,N_D=nd))
    return checks


def _write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _ranges(rows):
    accepted=[r["alpha"] for r in rows if r["passes"]]
    return [min(accepted),max(accepted)] if accepted else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-engine",action="store_true")
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    alphas=np.round(np.arange(-2.499,-1.0005,.001),3)
    scan=[evaluate(float(a)) for a in alphas]
    endpoints={w:[evaluate(float(a),optimize_windows=(w,)) for a in alphas] for w in previous.WINDOWS}
    rows=[]
    for r in scan:
        row={k:r[k] for k in ("alpha","A","score","passes","classification")}
        for window,ws in r["windows"].items():
            row[window+"_score"]=ws["score"]
            for mode,s in ws["modes"].items():row[window+"_"+mode+"_rate"]=s["rate"]
        row["D_med_Gpc"]=r["windows"]["conservative"]["modes"]["public"]["D_med_Gpc"]
        rows.append(row)
    with (OUTPUT/"ztf_scan.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUTPUT/"ztf_endpoint_scans.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=['window','alpha','A','score','passes']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for window,rs in endpoints.items():
            writer.writerows(dict(window=window,**{k:r[k] for k in fields[1:]}) for r in rs)
    chosen={"best_shared":min(scan,key=lambda r:r["score"])}
    for w,rs in endpoints.items():chosen["best_"+w]=min(rs,key=lambda r:r["score"])
    chosen["alpha_minus_two"]=evaluate(-2.)
    accepted=[r for r in scan if r["passes"]]
    if accepted:
        chosen["accepted_low_alpha"]=accepted[0];chosen["accepted_high_alpha"]=accepted[-1]
    highlights={};convergence=[]
    for label,r in chosen.items():
        low=evaluate(r["alpha"],r["A"],96,True,32)
        high=evaluate(r["alpha"],r["A"],192,True,64)
        coverage={str(c):{w:dict(score=previous.score(ws["modes"],c),
                 passes=previous.score(ws["modes"],c)<=1,
                 rates={m:s["raw_rate"]*c for m,s in ws["modes"].items()})
                 for w,ws in high["windows"].items()} for c in (.2,.35,.5)}
        highlights[label]=dict(result=high,coverage=coverage,
            optimization_target=label.removeprefix('best_') if label in
                ('best_conservative','best_optimistic') else 'shared',
            optimized_score=r['score'])
        for window in previous.WINDOWS:
            for mode in previous.MODE_KEYS:
                a=low["windows"][window]["modes"][mode];b=high["windows"][window]["modes"][mode]
                err=dict(rate=abs(a["rate"]/b["rate"]-1),q_median=abs(a["q_med"]-b["q_med"]),
                         L_median=abs(a["L_med_spectral"]/b["L_med_spectral"]-1),
                         D_median=abs(a["D_med_Gpc"]/b["D_med_Gpc"]-1),
                         independent_moment=b["independent_moment_relative_error"])
                convergence.append(dict(label=label,window=window,mode=mode,errors=err,
                                        passes=all(v<.01 for v in err.values())))
        print('highlight',label,r['alpha'],r['A'],r['score'],flush=True)
    _write(OUTPUT/"ztf_highlights.json",highlights)
    flat_highlights=[dict(label=label,alpha=value['result']['alpha'],A=value['result']['A'],
                         window=window,mode=mode,**stats)
                     for label,value in highlights.items()
                     for window,ws in value['result']['windows'].items()
                     for mode,stats in ws['modes'].items()]
    with (OUTPUT/"ztf_highlights.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat_highlights[0]));writer.writeheader();writer.writerows(flat_highlights)
    summary=dict(normalization=dict(L_ref=L_REF,A_units="Gpc^-3 yr^-1 per natural log L at L_ref"),
                 scan=dict(alpha_min=-2.499,alpha_max=-1.001,step=.001,count=len(scan)),
                 shared_accepted_alpha_range=_ranges(scan),
                 endpoint_accepted_alpha_ranges={w:_ranges(rs) for w,rs in endpoints.items()},
                 chosen={k:{x:v[x] for x in ("alpha","A","score")} for k,v in chosen.items()},
                 acceptance_neighbor_checks=[r for r in rows if accepted and
                    (abs(r['alpha']-accepted[0]['alpha'])<=.00101 or abs(r['alpha']-accepted[-1]['alpha'])<=.00101)],
                 convergence_boundary_checks=[rows[0],rows[-1]],
                 method="Analytic luminosity-distance and shared-phase moment integrals; segmented Gauss angular quadrature",
                 cut_timing="Rise/fade temporal exponents are selected at the peak, exactly as in the unchanged engine",
                 coverage=previous.COVERAGE,targets=previous.TARGETS,
                 physics=previous.PHYS_NEW,selection=previous.SELECTION,
                 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT/"analysis/lf_rate_medians.py",ROOT/"analysis/ztf_validation.py",
                             ROOT/"standalone_bridge.py",ROOT/"grb_detect/detection_rate.py",Path(__file__)]})
    _write(OUTPUT/"ztf_summary.json",summary)
    _write(OUTPUT/"ztf_verification.json",dict(resolution_doubling=convergence,engine_checks=[]))
    engine_checks=[]
    if not args.skip_engine:
        for label in ("best_shared","alpha_minus_two"):
            r=chosen[label]
            coarse=verify_engine(r['alpha'],r['A'])
            checks=verify_engine(r['alpha'],r['A'],4000,2000)
            for a,b in zip(coarse,checks):
                b['engine_resolution_change']=dict(rate=abs(a['engine_rate']/b['engine_rate']-1),
                    D_median=abs(a['engine_D_med_Gpc']/b['engine_D_med_Gpc']-1),
                    q_median=abs(a['engine_q_med']-b['engine_q_med']))
                b['passes']=b['passes'] and all(v<.01 for v in b['engine_resolution_change'].values())
            engine_checks.extend(checks)
            print('engine',label,'all_pass',all(c['passes'] for c in checks),flush=True)
            _write(OUTPUT/"ztf_verification.json",dict(resolution_doubling=convergence,engine_checks=engine_checks))
    print('completed_seconds',time.perf_counter()-start,flush=True)


if __name__ == "__main__":
    main()
