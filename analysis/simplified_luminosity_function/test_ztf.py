"""Independent checks of the all-L filtered ZTF reduction (no engine edits)."""
import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from analysis.simplified_luminosity_function import ztf


@pytest.mark.parametrize("mode",["public","high_cadence"])
@pytest.mark.parametrize("window",["conservative","optimistic"])
@pytest.mark.parametrize("alpha",[-2.4,-2.,-1.-1/.9,-1.-1/2.2,-1.746,-1.05])
def test_closed_joint_phase_moment_against_independent_quadrature(mode,window,alpha):
    case=ztf.build_case(mode,window)
    q=np.array([.3,1.,1.031,1.2,1.7,2.01,2.2,2.6])
    tp,wf,gamma,gw,rise,uc=ztf._profile(case,q)
    x,w=leggauss(160)
    total=np.zeros_like(q)
    for lo,hi in ((np.zeros_like(uc),uc),(uc,wf)):
        u=(lo[:,None]+hi[:,None])/2+(hi-lo)[:,None]*x/2
        direct=np.minimum(gw[:,None],rise[:,None]*(1+u/tp[:,None])**(-gamma[:,None]))
        total+=((hi-lo)/2)*(direct**(-alpha-1)@w)/case.mode['t_cad_s']
    np.testing.assert_allclose(ztf.filtered_moment(case,q,alpha),total,rtol=1e-10,atol=1e-20)


def test_conditional_luminosity_cdf_limits_and_join():
    for beta in (.05,.5,1.,1.4):
        threshold=np.array([1.,3.,9.])
        np.testing.assert_allclose(ztf.conditional_luminosity_cdf(threshold,threshold,beta),2*beta/3)
        np.testing.assert_allclose(ztf.conditional_luminosity_cdf(threshold*1e-100,threshold,beta),0,atol=1e-9)
        c=ztf.conditional_luminosity_cdf(np.logspace(-12,12,1000),1.,beta)
        assert np.all(np.diff(c)>=0)
        assert np.all((c>=0)&(c<=1))


def test_shared_distance_law_and_amplitude_scaling():
    one=ztf.evaluate(-1.746,22.,distributions=False)
    two=ztf.evaluate(-1.746,44.,distributions=False)
    medians=[]
    for window in one['windows']:
        for mode in one['windows'][window]['modes']:
            a=one['windows'][window]['modes'][mode]
            b=two['windows'][window]['modes'][mode]
            assert b['rate']==2*a['rate']
            medians.append(a['D_med_Gpc'])
    assert max(medians)==min(medians)
    assert medians[0]==pytest.approx(4.55*2**(-1/(2*(-1.746)+5)))


def test_minimax_normalization_centers_log_rate_errors():
    result=ztf.evaluate(-1.746)
    logs=[math.log(s['rate']/ztf.previous.TARGETS[mode]['rate'])
          for w in result['windows'].values() for mode,s in w['modes'].items()]
    assert max(logs)==pytest.approx(-min(logs),abs=1e-14)


def test_cutoff_expansion_removes_both_missing_tails():
    case=ztf.build_case('public','conservative')
    a=ztf.finite_statistics(case,-1.746,22.,26.,40.,64,24)
    b=ztf.finite_statistics(case,-1.746,22.,24.,42.,64,24)
    assert a['omitted_faint']>b['omitted_faint']>0
    assert a['omitted_bright']>b['omitted_bright']>=0
    assert b['retained_fraction']>a['retained_fraction']
    assert abs(sum([b['retained_fraction'],b['omitted_faint'],b['omitted_bright']])-1)<1e-14


def test_disallowed_slopes_rejected():
    case=ztf.build_case('public','conservative')
    for alpha in (-3.,-2.5,-1.,0.):
        with pytest.raises(ValueError):ztf.filtered_moment(case,np.array([1.]),alpha)


def test_reported_median_is_global_luminosity_cdf_median():
    case=ztf.build_case('high_cadence','optimistic')
    stats=ztf.statistics(case,-1.746,22.,64,24)
    threshold,mass,_=ztf.mixture(case,-1.746,64,24)
    assert ztf.mixture_cdf(stats['L_med_spectral'],threshold,mass,.746)==pytest.approx(.5,abs=1e-12)
