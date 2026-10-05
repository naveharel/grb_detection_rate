"""Unbounded differential luminosity rates and selected-population statistics.

The luminosity is L_nu(t_dec) at the observing band, in erg/s/Hz.  This
module integrates an event-rate intensity, never a normalized intrinsic PDF.
It retains the engine's light curve and selection rules (including its late
phase-III continuation).  Luminosity and radial integrations are analytic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .params import CM_TO_GPC
from .constants import DAY_S
from .survey import N_exp_max


@dataclass(frozen=True)
class PowerLawLuminosityFunction:
    alpha: float = -2.0
    log10_A: float = 0.0
    L_ref: float = 1e32

    def __post_init__(self):
        if not np.isfinite(self.alpha) or not -2.5 < self.alpha < -1.0:
            raise ValueError("The unbounded LF requires -2.5 < alpha < -1")
        if not np.isfinite(self.log10_A) or not -300 < self.log10_A < 300:
            raise ValueError("LF log10_A must be finite and between -300 and 300")
        if not np.isfinite(self.L_ref) or self.L_ref <= 0:
            raise ValueError("LF reference luminosity must be finite and positive")

    @property
    def A(self):
        """Rate density per natural log luminosity at L_ref [Gpc^-3 yr^-1]."""
        return 10.0 ** self.log10_A


def power_integral(lo, hi, exponent):
    """Integral of x**exponent, including convergent zero/infinite endpoints."""
    lo, hi, exponent = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float),
                                           np.asarray(exponent, float))
    out = np.zeros(lo.shape)
    good = hi > lo
    u = exponent + 1
    finite = good & (lo > 0) & np.isfinite(hi)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        z = np.log(np.where(finite, hi, 1)) - np.log(np.where(finite, lo, 1))
        # Factoring at the larger endpoint for u>0 avoids intermediate overflow.
        v = np.where(u > 0,
            np.exp(u*np.log(np.where(finite, hi, 1)))*(-np.expm1(-u*z))/np.where(u == 0, 1, u),
            np.exp(u*np.log(np.where(finite, lo, 1)))*np.expm1(u*z)/np.where(u == 0, 1, u))
        out = np.where(finite, np.where(np.abs(u) < 1e-13, z, v), out)
        left = good & (lo == 0) & np.isfinite(hi)
        right = good & (lo > 0) & np.isposinf(hi)
        out = np.where(left, np.where(u > 0, hi**u/u, np.inf), out)
        out = np.where(right, np.where(u < 0, -lo**u/u, np.inf), out)
        out = np.where(good & (lo == 0) & np.isposinf(hi), np.inf, out)
    return out


def _log_moment(coefficient, log_factor, lo, hi, exponent):
    """Signed coefficient times a power moment, evaluated entirely in logs."""
    coefficient,log_factor,lo,hi,exponent=np.broadcast_arrays(coefficient,log_factor,lo,hi,exponent)
    good=(coefficient!=0)&(hi>lo)&np.isfinite(log_factor)
    u=exponent+1
    result=np.zeros(coefficient.shape)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore",under="ignore"):
        width=hi-lo
        endpoint=np.where(u>0,hi,lo)
        log_integral=u*endpoint+np.log(-np.expm1(-np.abs(u)*width))-np.log(np.abs(u))
        log_integral=np.where(np.abs(u)<1e-13,np.log(width),log_integral)
        # The power must converge at each included infinite endpoint.
        divergent=(np.isneginf(lo)&(u<=0))|(np.isposinf(hi)&(u>=0))
        values=np.sign(coefficient)*np.exp(np.log(np.abs(coefficient))+log_factor+log_integral)
        values=np.where(divergent,np.sign(coefficient)*np.inf,values)
        result=np.where(good,values,0.)
    return result


def _log_rectangle_integral(alpha,lo,hi,c,log_b,w,log_d,v,q_floor,d_floor,differential=None):
    lo,hi,c,log_b,w,log_d,v,q_floor,d_floor=np.broadcast_arrays(
        lo,hi,c,log_b,w,log_d,v,q_floor,d_floor)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        q_cross=np.where(np.isfinite(log_b)&(q_floor>c),(np.log(np.maximum(q_floor-c,0))-log_b)/w,-np.inf)
        d_cross=np.where((v>0)&(d_floor>0),(np.log(d_floor)-log_d)/np.where(v>0,v,1),-np.inf)
        lower=np.maximum(lo,np.maximum(q_cross,d_cross))
        active=(np.isfinite(log_b)|(c>q_floor))&((v>0)|(log_d>np.log(d_floor)))&np.isfinite(log_d)
        if differential=="D":
            active=(np.isfinite(log_b)|(c>q_floor))&((v>0)|(log_d>=np.log(d_floor)))&np.isfinite(log_d)&(d_floor>0)
        upper=np.where(active,hi,lower)
        if differential=="q":
            value=2*q_floor*(_log_moment(1.,3*log_d,lower,upper,alpha+3*v)
                            -_log_moment(d_floor**3,0.,lower,upper,alpha))
        else:
            co=[c*c-q_floor*q_floor,2*c,np.ones(c.shape)]
            logs=[np.zeros(c.shape),log_b,2*log_b]
            if differential=="D":
                value=3*d_floor**2*sum(_log_moment(cf,lf,lower,upper,alpha+j*w)
                                      for j,(cf,lf) in enumerate(zip(co,logs)))
            else:
                value=sum(_log_moment(cf,lf+3*log_d,lower,upper,alpha+j*w+3*v)
                          -_log_moment(cf*d_floor**3,lf,lower,upper,alpha+j*w)
                          for j,(cf,lf) in enumerate(zip(co,logs)))
    return np.maximum(value,0.)


class RectangleDistribution:
    """Frozen most-contributing rectangles; logarithmic infinite LF moments."""
    def __init__(self,model,i,cadence,q_min,D_min,s_fade,s_rise,s_mode):
        self.model,self.cadence=model,np.asarray(cadence,float)
        self.alpha=model.lf.alpha
        self.q_min,self.d_min=float(q_min),float(D_min)/model.phys.D_euc_cm
        t=self.cadence;sh=t.shape;d=model.derived
        self.q_nr=float(d.q_nr)
        qi=model.q_i(i,t)
        fl=np.full(sh,d.F_dec_Jy)
        di=model.D_i(i,t,fl)/model.phys.D_euc_cm
        qs2,qs3=model._q_s_fading_caps(i,t,s_fade,s_mode)
        sj=np.full(sh,math.log(d.F_dec_Jy/d.F_j_Jy))
        snr=np.full(sh,math.log(d.F_dec_Jy/d.F_nr_Jy))
        with np.errstate(divide="ignore"):
            ldi=np.log(di)
        sqi=-2*ldi
        w2,w3=1/(2*model.pls.a_II(model.phys.p)),1/(2*model.pls.a_III(model.phys.p))
        rise=s_rise>0
        log_eta=math.log(10)*s_rise*t/(2.5*DAY_S) if rise else np.zeros(sh)
        lise=-log_eta/2
        srj=log_eta+sj
        srnr=log_eta+snr
        if rise:
            arg=log_eta+2*ldi
            lqt=np.where(arg < -sj,(1-2*w3)*math.log(d.q_dec-1)-w3*arg,
                          math.log(d.q_dec-1)-w2*arg)
            qri=1+np.exp(np.minimum(lqt,700))
        else:qri=np.full(sh,np.inf)
        ld7=np.minimum(np.minimum(0,ldi),lise)
        qc=[np.minimum(d.q_nr,qs3),qs3,qs2,np.minimum(np.minimum(d.q_nr,qs3),qri),
            np.minimum(np.minimum(qi,qs3),qri),np.minimum(np.minimum(qi,qs2),qri),
            np.minimum(d.q_dec,qs2)]
        qc=[np.minimum(x,d.q_nr) for x in qc]
        def atq(q):
            q=np.asarray(q,float)
            with np.errstate(divide="ignore",invalid="ignore"):
                result=np.where(q<2,log_eta+np.log((q-1)/(d.q_dec-1))/w2,
                                srj+np.log(q-1)/w3)
            return np.where(q<=1,-np.inf,result)
        ldm=math.log(self.d_min) if self.d_min>0 else -np.inf
        edges=[np.zeros(sh),sj,snr,sqi,atq(qs2),atq(qs3),atq(self.q_min),
               2*(ldm-ldi),2*(ldm-ld7)]
        if rise:edges += [log_eta,srj,srnr,-2*lise,2*(ldm-lise),2*(ldm-np.minimum(ldi,lise))]
        edges=np.stack([np.broadcast_to(x,sh) for x in edges])
        edges=np.where(np.isnan(edges),np.inf,edges)
        edges=np.sort(np.concatenate([np.full((1,)+sh,-np.inf),edges,np.full((1,)+sh,np.inf)]),axis=0)
        self.segments=[];self.regime_mass=np.zeros((7,)+sh)
        for lo,hi in zip(edges[:-1],edges[1:]):
            valid_segment=hi>lo
            with np.errstate(invalid="ignore"):
                mid=np.where(np.isneginf(lo),hi-1,np.where(np.isposinf(hi),lo+1,(lo+hi)/2))
            mid=np.where(valid_segment,mid,0)
            qe=1+np.exp(np.minimum(np.where(mid<=sj,math.log(d.q_dec-1)+w2*mid,w3*(mid-sj)),700))
            fam=qe>=qi
            rid=np.select([fam&(qe>=d.q_nr),fam&(qe>=2),fam&(qe>=d.q_dec),fam,
                           qi>=d.q_nr,qi>=2,qi>=d.q_dec],[1,2,3,7,4,5,6],default=7)
            small=mid<=srj
            lb=np.where(small,math.log(d.q_dec-1)-w2*log_eta,-w3*srj)
            w=np.where(small,w2,w3)
            qe_r=1+np.exp(np.minimum(lb+w*mid,700))
            cap=np.choose(rid-1,qc)
            flux=(rid<=3)&(qe_r<cap)
            c=np.where(flux,1,cap);lb=np.where(flux,lb,-np.inf)
            v=np.where(rid<=3,0.,.5)
            ld=np.where(rid<=3,0.,np.where(rid<=6,ldi,ld7))
            valid=np.where(rid<=3,mid>=log_eta,np.where(rid<=6,qri>=d.q_dec,True)) if rise else np.ones(sh,bool)
            ld=np.where(valid&valid_segment,ld,-np.inf)
            angular=_log_rectangle_integral(self.alpha,lo,hi,c,lb,w,ld,v,self.q_min,self.d_min)
            if rise:
                on_d=np.where(rid<=3,np.where(mid<log_eta,lise,0.),np.minimum(ldi,lise))
                on_v=np.where((rid<=3)&(mid>=log_eta),0.,.5)
                on_d=np.where((rid!=7)&valid_segment,on_d,-np.inf)
                on=_log_rectangle_integral(self.alpha,lo,hi,qc[6],-np.inf,w,on_d,on_v,self.q_min,self.d_min)
                pick=on>angular
                c,lb,ld,v=np.where(pick,qc[6],c),np.where(pick,-np.inf,lb),np.where(pick,on_d,ld),np.where(pick,on_v,v)
                angular=np.maximum(angular,on)
            self.segments.append((lo,hi,c,lb,w,ld,v))
            for k in range(7):self.regime_mass[k]+=np.where(rid==k+1,angular,0)
        self.total=self.regime_mass.sum(axis=0)

    def survival(self,q=None,distance=None,differential=None):
        q=self.q_min if q is None else np.maximum(q,self.q_min)
        distance=self.d_min if distance is None else np.maximum(distance,self.d_min)
        return np.maximum(sum(_log_rectangle_integral(self.alpha,*s,q,distance,differential)
                              for s in self.segments),0.)

    def medians(self):
        target=self.total/2
        lo,hi=np.full(self.total.shape,self.q_min),np.full(self.total.shape,self.q_nr)
        for _ in range(38):
            mid=(lo+hi)/2;below=self.survival(q=mid)>target
            lo,hi=np.where(below,mid,lo),np.where(below,hi,mid)
        qm=(lo+hi)/2
        dlo=np.full(self.total.shape,math.log(max(self.d_min,1e-300)));dhi=np.zeros(self.total.shape)
        for _ in range(48):
            mid=(dlo+dhi)/2;below=self.survival(distance=np.exp(mid))>target
            dlo,dhi=np.where(below,mid,dlo),np.where(below,dhi,mid)
        dm=np.exp((dlo+dhi)/2)
        return np.where(self.total>0,qm,np.nan),np.where(self.total>0,dm,np.nan)


# Explicit accuracy control for convergence checks; N_q remains a compatibility
# argument in the public legacy methods, while this path uses adaptive panels.
ANGULAR_RTOL = 2e-10
_GL = {n:np.polynomial.legendre.leggauss(n) for n in (8,16)}


def _exact_density(model,i,q,t,s_fade,s_rise,s_mode,rise_random_start,fade_random_start):
    """q times the shared-phase flux-shape moment, with engine selection."""
    q,t = np.broadcast_arrays(np.asarray(q,float),np.asarray(t,float))
    d = model.derived
    qt = np.maximum(q-1,1e-30)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        peak = np.where(q<d.q_dec,1.,np.where(q<2,
            (qt/(d.q_dec-1))**(-2*model.pls.a_II(model.phys.p)),
            (d.q_dec-1)**-2*(qt/(d.q_dec-1))**(-2*model.pls.a_III(model.phys.p))))
        fl = np.full(np.broadcast(q,t).shape,d.F_dec_Jy)
        if model.win_from_peak:
            win = (model._D_eff_window_cm(q,i,t,fl)/model.phys.D_euc_cm)**2
        else:
            win = (model.D_i(i,t,fl)/model.phys.D_euc_cm)**2
        horizon = np.minimum(peak,win)
        beta = -model.lf.alpha-1
        if s_rise <= 0:
            return q*horizon**beta*model._fading_survival(q,i,t,s_fade,s_mode,
                                                     fade_random_start=fade_random_start)
        tp,tf,k,ise = model._rise_fade_windows(q,i,t,s_fade,s_rise,s_mode)
        if not fade_random_start:
            tf = np.where(tf>=tp,np.inf,-np.inf)
        width = np.clip(tf-tp,0,t)
        log_rise = np.log(peak)-math.log(10)*float(s_rise)*t/(2.5*DAY_S)
        log_horizon = np.log(horizon)
        if not rise_random_start:
            return q*width/t*np.exp(beta*np.minimum(log_horizon,log_rise))
        gamma = 2/k
        active = (horizon>0)&(width>0)
        logratio = (log_rise-log_horizon)/gamma
        switch = np.clip(tp*np.expm1(np.minimum(logratio,700)),0,width)
        z = 1-gamma*beta
        logr = np.log1p((width-switch)/(tp+switch))
        primitive = np.where(np.abs(z)>1e-11,np.expm1(z*logr)/np.where(z==0,1,z),logr)
        ramp = np.exp(beta*log_rise)*tp*(1+switch/tp)**z*primitive
        moment = (switch*horizon**beta+ramp)/t
        return np.where(active,q*np.maximum(moment,0),0.)


class AngularDistribution:
    """Adaptive Gaussian angle integral with analytic radial distribution."""
    def __init__(self,model,i,cadence,q_min,D_min,s_fade,s_rise,s_mode,
                 rise_random_start,fade_random_start):
        self.model,self.i,self.cadence = model,i,np.asarray(cadence,float)
        self.q_min,self.d_min = float(q_min),float(D_min)/model.phys.D_euc_cm
        self.options = (s_fade,s_rise,s_mode,rise_random_start,fade_random_start)
        self.h = 2*model.lf.alpha+5
        self.beta = -model.lf.alpha-1
        n = len(self.cadence)
        boundaries = [np.full(n,x) for x in (q_min,model.derived.q_dec,2.,model.derived.q_nr)]
        def q_at_time(time):
            time=np.maximum(time,model.derived.t_dec_s)
            return 1+(time/model.derived.t_j_s)**np.where(time<model.derived.t_j_s,3/8,1/2)
        req=model._t_req_s(i,self.cadence)
        boundaries.append(q_at_time(model.derived.t_j_s-req) if model.win_from_peak
                          else model.q_i(i,self.cadence))
        if s_fade>0:
            for tf in model._t_first_caps(i,self.cadence,s_fade,s_mode):
                boundaries.extend([q_at_time(tf),q_at_time(tf-self.cadence)])
        boundaries=np.sort(np.clip(np.stack(boundaries),min(q_min,model.derived.q_nr),model.derived.q_nr),axis=0)
        self.left,self.right,self.cells,self.mass = [],[],[],[]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            keep=b>a
            left,right,cells = a[keep],b[keep],np.arange(n)[keep]
            for depth in range(22):
                if not len(cells):break
                fine = self.integral(left,right,cells,16)
                coarse = self.integral(left,right,cells,8)
                accept = (np.abs(fine-coarse) <= ANGULAR_RTOL*np.maximum(abs(fine),1e-280)) | (depth==21)
                self.left.extend(left[accept]);self.right.extend(right[accept])
                self.cells.extend(cells[accept]);self.mass.extend(fine[accept])
                left,right,cells = left[~accept],right[~accept],cells[~accept]
                middle=(left+right)/2
                left,right,cells=np.r_[left,middle],np.r_[middle,right],np.r_[cells,cells]
        self.left,self.right,self.cells,self.mass = (np.asarray(self.left),np.asarray(self.right),
                                                   np.asarray(self.cells,int),np.asarray(self.mass))
        self.total = np.bincount(self.cells,weights=self.mass,minlength=n)
        self.radial = -np.expm1(self.h*np.log(self.d_min))/self.h if 0<self.d_min<1 else (1/self.h if self.d_min<=0 else 0.)

    def integral(self,left,right,cells,n=16):
        x,w = _GL[n]
        q = (left+right)[None,:]/2+(right-left)[None,:]*x[:,None]/2
        values = _exact_density(self.model,self.i,q,self.cadence[cells][None,:],*self.options)
        return (right-left)/2*np.sum(w[:,None]*values,axis=0)

    def survival(self,q):
        q = np.broadcast_to(q,self.total.shape)
        lower = np.minimum(self.right,np.maximum(self.left,q[self.cells]))
        masses = self.integral(lower,self.right,self.cells)
        return np.bincount(self.cells,weights=masses,minlength=len(self.total))

    def medians(self):
        lo,hi = np.full(self.total.shape,self.q_min),np.full(self.total.shape,self.model.derived.q_nr)
        for _ in range(38):
            mid=(lo+hi)/2
            below=self.survival(mid)>self.total/2
            lo,hi=np.where(below,mid,lo),np.where(below,hi,mid)
        qm=(lo+hi)/2
        if self.d_min<=0:
            dm=math.exp(-math.log(2)/self.h)
        elif self.d_min>=1:
            dm=np.nan
        else:
            dm=math.exp(np.logaddexp(0,self.h*math.log(self.d_min))/self.h-math.log(2)/self.h)
        ok=(self.total>0)&(self.radial>0)
        return np.where(ok,qm,np.nan),np.where(ok,dm,np.nan)


def _prepared(model,i,cadence,full_integral,**options):
    cadence=np.asarray(cadence,float).reshape(-1)
    d=model.derived
    shape=(model.lf.alpha,model.win_from_peak,model.win_i_minus_one,
           model.phys.p,model.phys.theta_j_rad,model.phys.D_euc_cm,
           model.pls.a_II(model.phys.p),model.pls.a_III(model.phys.p),
           model.pls.alpha_II_temporal(model.phys.p),model.pls.alpha_III_temporal(model.phys.p),
           d.t_dec_s,d.t_j_s,d.q_dec,d.q_nr,d.F_j_Jy/d.F_dec_Jy,d.F_nr_Jy/d.F_dec_Jy)
    key=(shape,i,tuple(cadence),bool(full_integral),ANGULAR_RTOL,tuple(options.items()))
    cache=getattr(model,"_scale_free_cache",None)
    if cache is None:model._scale_free_cache=cache={}
    if key not in cache:
        if len(cache)>=12:cache.pop(next(iter(cache)))
        cls=AngularDistribution if full_integral else RectangleDistribution
        kwargs=dict(options)
        if not full_integral:
            kwargs.pop("rise_random_start");kwargs.pop("fade_random_start")
        cache[key]=cls(model,i,cadence,**kwargs)
    return cache[key]


def _options(q_min=0.,D_min_cm=0.,s_fade=0.,s_rise=0.,s_mode="discrete",
             rise_random_start=True,fade_random_start=True):
    return dict(q_min=max(0.,float(q_min)),D_min=max(0.,float(D_min_cm)),s_fade=float(s_fade),
                s_rise=float(s_rise),s_mode=s_mode,rise_random_start=bool(rise_random_start),
                fade_random_start=bool(fade_random_start))


def _strategy(model,N,t):
    N,t=np.broadcast_arrays(np.asarray(N,float),np.asarray(t,float))
    exposure=model.t_exp_s(N,t)
    valid=(N>=1-1e-12)&(N<=N_exp_max(model.instrument)+1e-12)&np.isfinite(exposure)&(exposure>0)&(t>0)
    flux=model.F_lim_Jy(exposure)
    L_dec=4*np.pi*model.phys.D_euc_cm**2*flux*1e-23
    volume=4*np.pi/3*(model.phys.D_euc_cm*CM_TO_GPC)**3
    factor=model.f_Omega(N)*model.phys.theta_j_rad**2*volume*model.lf.A*(L_dec/model.lf.L_ref)**(model.lf.alpha+1)
    return N,t,valid,factor


def rate(model,i,N,t,*,full_integral=False,return_regime=False,**kwargs):
    N,t,valid,factor=_strategy(model,N,t)
    answer=np.full(N.shape,np.nan);regime=np.full(N.shape,np.nan)
    if np.any(valid):
        times,inverse=np.unique(t[valid],return_inverse=True)
        prepared=_prepared(model,i,times,full_integral,**_options(**kwargs))
        if full_integral:
            mass=3/prepared.beta*prepared.radial*prepared.total
        else:
            mass=.5*prepared.total
            regime[valid]=(np.argmax(prepared.regime_mass,axis=0)+1)[inverse]
        answer[valid]=factor[valid]*mass[inverse]
        regime=np.where(answer>0,regime,np.nan)
    return (answer,regime) if return_regime else answer


def log_rate(value):
    """Distinguish a physical zero (-inf) from an invalid strategy (NaN)."""
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.log10(value)


def medians(model,i,N,t,*,full_integral=False,**kwargs):
    N,t,valid,_=_strategy(model,N,t)
    qm,dm=np.full(N.shape,np.nan),np.full(N.shape,np.nan)
    if np.any(valid):
        times,inverse=np.unique(t[valid],return_inverse=True)
        prepared=_prepared(model,i,times,full_integral,**_options(**kwargs))
        if not hasattr(prepared,"_medians"):prepared._medians=prepared.medians()
        q,d=prepared._medians
        qm[valid],dm[valid]=q[inverse],d[inverse]*model.phys.D_euc_cm
    return qm,dm


def distributions(model,i,N,t,*,full_integral=False,q_values=None,D_values_cm=None,**kwargs):
    """Scalar-strategy selected-mode survival functions and analytic densities."""
    _,_,valid,factor=_strategy(model,N,t)
    if np.ndim(valid):raise ValueError("LF distributions require a scalar strategy")
    q=np.asarray([] if q_values is None else q_values,float)
    D=np.asarray([] if D_values_cm is None else D_values_cm,float)
    opts=_options(**kwargs);q0=opts["q_min"];d0=opts["D_min"]/model.phys.D_euc_cm
    empty=dict(total_rate=np.nan,q_survival=np.full(q.shape,np.nan),dR_dq=np.full(q.shape,np.nan),
               D_survival=np.full(D.shape,np.nan),dR_dD_per_cm=np.full(D.shape,np.nan),q_med=np.nan,D_med_cm=np.nan)
    if not valid:return empty
    dist=_prepared(model,i,np.array([t]),full_integral,**opts)
    qmed,dmed=medians(model,i,N,t,full_integral=full_integral,**kwargs)
    if full_integral:
        factor=float(factor)*3/dist.beta
        total=factor*dist.radial*float(dist.total[0])
        qs=np.array([factor*dist.radial*dist.survival(np.array([v]))[0] for v in q.flat]).reshape(q.shape)
        qd=factor*dist.radial*_exact_density(model,i,q,np.asarray(t),*dist.options)
        qd=np.where((q>=q0)&(q<=model.derived.q_nr),qd,0.)
        dn=np.clip(D/model.phys.D_euc_cm,max(0.,min(d0,1)),1)
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            ds=factor*float(dist.total[0])*(-np.expm1(dist.h*np.log(dn)))/dist.h
            dd=factor*float(dist.total[0])*dn**(dist.h-1)/model.phys.D_euc_cm
        dd=np.where((D>=opts["D_min"])&(D>0)&(D<=model.phys.D_euc_cm)&(d0<1),dd,0.)
        ds=np.where(d0>=1,0.,ds)
    else:
        factor=float(factor)/2
        total=factor*float(dist.total[0])
        qs=factor*dist.survival(q=q[...,None])[...,0]
        qd=factor*dist.survival(q=q[...,None],differential="q")[...,0]
        qd=np.where((q>=q0)&(q<=model.derived.q_nr),qd,0.)
        ds=factor*dist.survival(distance=(D/model.phys.D_euc_cm)[...,None])[...,0]
        dd=factor*dist.survival(distance=(D/model.phys.D_euc_cm)[...,None],differential="D")[...,0]/model.phys.D_euc_cm
        dd=np.where((D>=opts["D_min"])&(D>0)&(D<=model.phys.D_euc_cm),dd,0.)
    return dict(total_rate=total,q_survival=qs,dR_dq=qd,D_survival=ds,dR_dD_per_cm=dd,
                q_med=float(qmed),D_med_cm=float(dmed))
