"""Independent checks of the manuscript's expanded physical-luminosity formulas.

The expanded expressions and the finite-bound direct quadrature below are
implemented here independently of the production theory helpers. No engine is
changed or used. Run with pytest or unittest from the repository root.
"""
import math
import unittest

import numpy as np

from analysis.simplified_luminosity_function import theory


def _difference_of_powers(upper, lower, exponent):
    """Explicit quotient in the manuscript, including its removable log limit."""
    if abs(exponent) < 1e-12:
        return math.log(upper/lower)
    return (upper**exponent-lower**exponent)/exponent


def _expanded_physical_luminosity_integral(alpha, shape, L_dec, L_ref, required_time=None):
    """Integral q [L_Euc(max(q,q_i))/L_ref]**(alpha+1) dq, expanded in full."""
    L_j=L_dec*(shape.gamma0*shape.theta_j)**(2*(shape.p-1))
    if required_time is None or required_time <= shape.t_dec_s:
        q_i=shape.q_dec
        L_i=L_dec
    elif required_time < shape.t_j_s:
        q_i=1+(required_time/shape.t_j_s)**(3/8)
        L_i=L_dec*(required_time/shape.t_dec_s)**(3*(shape.p-1)/4)
    elif required_time < shape.t_nr_s:
        q_i=1+(required_time/shape.t_j_s)**.5
        L_i=L_j*(required_time/shape.t_j_s)**shape.p
    else:
        q_i=shape.q_nr
        L_nr=L_j*(shape.q_nr-1)**(2*shape.p)
        L_i=L_nr*(required_time/shape.t_nr_s)**((15*shape.p-21)/10)
    total=q_i**2/2*(L_i/L_ref)**(alpha+1)
    if q_i < 2:
        total+=(L_j/L_ref)**(alpha+1)*(
            _difference_of_powers(1,q_i-1,2+2*(shape.p-1)*(alpha+1))
            +_difference_of_powers(1,q_i-1,1+2*(shape.p-1)*(alpha+1)))
    if q_i < shape.q_nr:
        total+=(L_j/L_ref)**(alpha+1)*(
            _difference_of_powers(shape.q_nr-1,max(q_i,2)-1,2+2*shape.p*(alpha+1))
            +_difference_of_powers(shape.q_nr-1,max(q_i,2)-1,1+2*shape.p*(alpha+1)))
    return total


def _direct_gauss(fn, lower, upper, nodes, weights):
    if upper <= lower:
        return 0.
    return (upper-lower)/2*float(np.dot(weights,fn(
        (upper+lower)/2+(upper-lower)*nodes/2)))


def _expanded_angular_volume(luminosity, L_dec, shape):
    """Manuscript's accessible-angle volume, using explicit physical thresholds."""
    L_j=L_dec*(shape.gamma0*shape.theta_j)**(2*(shape.p-1))
    L_nr=L_j*(shape.q_nr-1)**(2*shape.p)
    if luminosity >= L_nr:
        return shape.q_nr**2/2
    if luminosity < L_dec:
        return (luminosity/L_dec)**1.5*_expanded_angular_volume(L_dec,L_dec,shape)
    if luminosity < L_j:
        q_Euc=1+(luminosity/L_j)**(1/(2*(shape.p-1)))
        unsaturated=(
            _difference_of_powers(1,q_Euc-1,2-3*(shape.p-1))
            +_difference_of_powers(1,q_Euc-1,1-3*(shape.p-1))
            +_difference_of_powers(shape.q_nr-1,1,2-3*shape.p)
            +_difference_of_powers(shape.q_nr-1,1,1-3*shape.p))
    else:
        q_Euc=1+(luminosity/L_j)**(1/(2*shape.p))
        unsaturated=(
            _difference_of_powers(shape.q_nr-1,q_Euc-1,2-3*shape.p)
            +_difference_of_powers(shape.q_nr-1,q_Euc-1,1-3*shape.p))
    return q_Euc**2/2+(luminosity/L_j)**1.5*unsaturated


def _finite_rate_and_unsaturated_part(alpha, L_min, L_max, flux_factor, shape):
    """Direct double quadrature at fixed differential LF amplitude.

    Physical luminosities are integrated in log L. The flux factor multiplies
    L_dec, while LF bounds and its fixed physical reference remain unchanged.
    Common sky, angle, volume, and amplitude factors cancel in the response.
    """
    nodes,weights=np.polynomial.legendre.leggauss(64)
    L_dec=1e30*flux_factor
    L_j=L_dec*(shape.gamma0*shape.theta_j)**(2*(shape.p-1))
    L_ref=1e32
    bounds=[0.,shape.q_dec,2.,shape.q_nr]
    for luminosity in (L_min,L_max):
        if luminosity >= L_dec:
            if luminosity < L_j:
                q=1+(luminosity/L_j)**(1/(2*(shape.p-1)))
            else:
                q=1+(luminosity/L_j)**(1/(2*shape.p))
            if shape.q_dec < q < shape.q_nr:
                bounds.append(q)
    bounds=sorted(set(bounds))
    total=0.
    unsaturated=0.
    for lower,upper in zip(bounds[:-1],bounds[1:]):
        q_values=(lower+upper)/2+(upper-lower)*nodes/2
        rate_values=[]
        unsaturated_values=[]
        for q in q_values:
            if q <= shape.q_dec:
                L_Euc=L_dec
            elif q < 2:
                L_Euc=L_j*(q-1)**(2*(shape.p-1))
            else:
                L_Euc=L_j*(q-1)**(2*shape.p)
            def luminosity_density(logL):
                luminosity=np.exp(logL)
                return (luminosity/L_ref)**(alpha+1)*np.minimum(
                    (luminosity/L_Euc)**1.5,1.)
            below=_direct_gauss(luminosity_density,math.log(L_min),
                math.log(min(L_max,max(L_min,L_Euc))),nodes,weights)
            above=_direct_gauss(luminosity_density,
                math.log(max(L_min,min(L_max,L_Euc))),math.log(L_max),nodes,weights)
            rate_values.append(q*(below+above))
            unsaturated_values.append(q*below)
        total+=(upper-lower)/2*float(np.dot(weights,rate_values))
        unsaturated+=(upper-lower)/2*float(np.dot(weights,unsaturated_values))
    return total,unsaturated


class PedagogicalEquationTests(unittest.TestCase):
    def test_global_angular_volume_against_direct_quadrature_and_cdf_derivative(self):
        shape=theory.ModelShape()
        nodes,weights=np.polynomial.legendre.leggauss(96)
        L_dec=1e30
        L_j=L_dec*(shape.gamma0*shape.theta_j)**(2*(shape.p-1))
        L_nr=L_j*(shape.q_nr-1)**(2*shape.p)
        luminosities=[.01*L_dec,L_dec,2*L_dec,math.sqrt(L_dec*L_j),
                      L_j,2*L_j,math.sqrt(L_j*L_nr),.99*L_nr,L_nr,100*L_nr]
        for luminosity in luminosities:
            cuts=[0.,shape.q_dec,2.,shape.q_nr]
            if L_dec < luminosity < L_nr:
                # Locate the volume kink independently using the light-curve
                # peak, rather than the analytic q_Euc inversion being tested.
                lower,upper=shape.q_dec,shape.q_nr
                for _ in range(60):
                    middle=(lower+upper)/2
                    if luminosity*float(shape.peak_shape(middle)) > L_dec:
                        lower=middle
                    else:
                        upper=middle
                cuts.append((lower+upper)/2)
            cuts=sorted(set(cuts))
            direct=sum(_direct_gauss(lambda q:q*np.minimum(
                (luminosity*shape.peak_shape(q)/L_dec)**1.5,1.),
                lower,upper,nodes,weights) for lower,upper in zip(cuts[:-1],cuts[1:]))
            explicit=_expanded_angular_volume(luminosity,L_dec,shape)
            self.assertLess(abs(explicit/direct-1),2e-12)
            for alpha in (-2.4,-2.,-1.25):
                # At L_dec the core has a genuine corner in the density's
                # derivative, so use a small step across that threshold.
                step=1e-6 if luminosity == L_dec else 1e-4
                derivative=(theory.luminosity_cdf(luminosity/L_dec*math.exp(step),alpha)
                            -theory.luminosity_cdf(luminosity/L_dec*math.exp(-step),alpha))/(2*step)
                expected=(-alpha-1)*(2*alpha+5)/3*(luminosity/L_dec)**(alpha+1)*explicit/theory.angular_moment(alpha)
                np.testing.assert_allclose(derivative,expected,rtol=2e-6,atol=2e-10)

    def test_global_intermediate_slope_in_a_controlled_asymptotic_limit(self):
        # Widen the separation of q=1, q_Euc, q_nr so the manuscript's
        # asymptote is testable; the default q_nr is only about 14.
        shape=theory.ModelShape(theta_j=1e-9,gamma0=10**1.5/1e-9)
        L_dec=1e30
        L_j=L_dec*(shape.gamma0*shape.theta_j)**(2*(shape.p-1))
        step=1e-3
        for q_Euc in (1e4,1e5):
            luminosity=L_j*(q_Euc-1)**(2*shape.p)
            for alpha in (-2.,-1.4,-1.25):
                lower=luminosity*math.exp(-step)
                upper=luminosity*math.exp(step)
                measured=(math.log(_expanded_angular_volume(upper,L_dec,shape))
                          -math.log(_expanded_angular_volume(lower,L_dec,shape)))/(2*step)+alpha+1
                self.assertAlmostEqual(measured,alpha+1+1/shape.p,delta=4e-5)

    def test_expanded_angular_expression_and_all_four_log_limits(self):
        shape=theory.ModelShape()
        slopes=[-2.4,-2,-1.75,-1.25,-1.05,
                -1-1/(shape.p-1),-1-1/(2*(shape.p-1)),
                -1-1/shape.p,-1-1/(2*shape.p)]
        for alpha in slopes:
            for required_time in (None,.5*shape.t_dec_s,.4*shape.t_j_s,
                                  shape.t_j_s,4*shape.t_j_s,2*shape.t_nr_s):
                for L_dec in (1e29,1e32):
                    result=_expanded_physical_luminosity_integral(alpha,shape,L_dec,1e32,required_time)
                    selection=(dict(prescription="continuous") if required_time is None else
                               dict(prescription="legacy",i_det=2,cadence_s=required_time/2))
                    reference=(L_dec/1e32)**(alpha+1)*theory.angular_moment(alpha,shape=shape,**selection)
                    self.assertLess(abs(result/reference-1),1e-10)

    def test_finite_cutoff_flux_response_is_unsaturated_fraction(self):
        shape=theory.ModelShape()
        step=1e-5
        for alpha in (-3.,-2.,-1.,-.4):
            for L_min,L_max in ((1e26,1e28),(1e29,1e38),(1e41,1e43)):
                for flux_factor in (.3,1.,4.):
                    rate,unsaturated=_finite_rate_and_unsaturated_part(
                        alpha,L_min,L_max,flux_factor,shape)
                    lower,_=_finite_rate_and_unsaturated_part(
                        alpha,L_min,L_max,flux_factor*math.exp(-step),shape)
                    upper,_=_finite_rate_and_unsaturated_part(
                        alpha,L_min,L_max,flux_factor*math.exp(step),shape)
                    derivative=(math.log(upper)-math.log(lower))/(2*step)
                    np.testing.assert_allclose(derivative,-1.5*unsaturated/rate,
                                               rtol=1e-7,atol=1e-9)

    def test_fixed_angle_cdf_and_median_without_luminosity_rescaling_shorthand(self):
        nodes,weights=np.polynomial.legendre.leggauss(128)
        for alpha in (-2.4,-2.,-1.75,-1.25,-1.05):
            for L_Euc in (1e30,1e35):
                if alpha <= -7/4:
                    median=L_Euc*(3/(4*(-alpha-1)))**(1/(alpha+2.5))
                else:
                    median=L_Euc*(4*(alpha+2.5)/3)**(-1/(alpha+1))
                # Integrate the normalized log-L density independently,
                # extending the remaining faint tail analytically.
                lower=math.log(L_Euc)-800
                upper=math.log(median)
                boundary=math.log(L_Euc)
                def density(logL):
                    return 2*(-alpha-1)*(alpha+2.5)/3*np.exp(np.where(
                        logL<=boundary,(alpha+2.5)*(logL-boundary),
                        (alpha+1)*(logL-boundary)))
                cuts=sorted(set([lower,upper]+([boundary] if lower<boundary<upper else [])))
                cdf=sum(_direct_gauss(density,a,b,nodes,weights)
                        for a,b in zip(cuts[:-1],cuts[1:]))
                cdf+=2*(-alpha-1)/3*math.exp((alpha+2.5)*(lower-boundary))
                self.assertAlmostEqual(cdf,.5,places=9)
                self.assertAlmostEqual(float(theory.conditional_luminosity_cdf(median/L_Euc,alpha)),.5,places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
