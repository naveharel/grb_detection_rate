"""Independent numerical checks for the standalone scale-free LF derivation."""
import math
import unittest

import numpy as np

from analysis.simplified_luminosity_function import theory as th


class TheoryTests(unittest.TestCase):
    def assertRelative(self, actual, expected, tolerance=1e-7):
        self.assertLess(abs(actual/expected-1), tolerance)

    def test_piecewise_lightcurve_continuity_and_newtonian_slope(self):
        s = th.ModelShape()
        for t in (s.t_dec_s, s.t_j_s, s.t_nr_s):
            self.assertRelative(s.lightcurve(t*(1-1e-10)), s.lightcurve(t*(1+1e-10)), 1e-8)
        self.assertRelative(s.lightcurve(2*s.t_nr_s)/s.lightcurve(s.t_nr_s),
                            2**((21-15*s.p)/10))
        self.assertRelative(s.lightcurve(2*s.t_nr_s, False)/s.lightcurve(s.t_nr_s), 2**(-s.p))

    def test_angular_closed_form_against_independent_quadrature_and_log_cases(self):
        # Includes m=1 and m=2 in each angular phase, requiring log primitives.
        slopes = [-2.4, -2, -1.75, -1.25, -1.05, -4/3, -5/3, -1.2, -1.4]
        for alpha in slopes:
            for prescription, cadence, count in [("continuous", None, 2),
                ("legacy", .2*th.DAY_S, 2), ("legacy", 2*th.DAY_S, 2),
                ("legacy", 2*th.DAY_S, 10), ("legacy", th.DAY_S, 600)]:
                kw = dict(prescription=prescription, cadence_s=cadence, i_det=count)
                self.assertRelative(th.angular_moment(alpha, **kw),
                    th.angular_moment(alpha, method="quadrature", n=256, **kw))

    def test_time_primitive_against_lightcurve_quadrature_including_logs(self):
        s = th.ModelShape()
        for b in [.05, .25, 1, 1.4, 1/s.a_II, 1/s.p, 1/s.a_IV]:
            for lo, hi in [(0, 2*s.t_dec_s), (.1*s.t_j_s, 3*s.t_j_s),
                           (.1*s.t_nr_s, 10*s.t_nr_s)]:
                bounds = sorted(set([lo, hi]+[t for t in [s.t_dec_s,s.t_j_s,s.t_nr_s] if lo<t<hi]))
                numeric = th.gauss_integral(lambda t:s.lightcurve(t)**b, bounds, 128)
                self.assertRelative(s.time_moment(lo, hi, b), numeric)

    def test_phase_average_independent_two_dimensional_quadrature(self):
        s = th.ModelShape()
        for alpha, cadence, count in [(-2, 2*th.DAY_S, 2),
            (-1.25, 2*th.DAY_S, 10), (-1.75, th.DAY_S, 300)]:
            b = -alpha-1
            def density(q_values):
                out = []
                for q in q_values:
                    lo = float(s.peak_time(q))+(count-1)*cadence
                    hi = lo+cadence
                    bounds = [lo,hi]+[t for t in (s.t_j_s,s.t_nr_s) if lo<t<hi]
                    out.append(q*th.gauss_integral(lambda t:s.lightcurve(t)**b, bounds, 96)/cadence)
                return np.array(out)
            qbounds = th.q_bounds(s, cadence, count, "random_phase")
            numeric = th.gauss_integral(density, qbounds, 96)
            self.assertRelative(th.phase_averaged_moment(alpha, cadence, count), numeric)

    def test_single_visit_random_phase_and_continuous_limit(self):
        s=th.ModelShape()
        alpha,cadence=-2.,th.DAY_S
        b=-alpha-1
        def density(q_values):
            out=[]
            for q in q_values:
                lo=float(s.peak_time(q))
                hi=lo+cadence
                # Log-time quadrature independently resolves the short on-axis
                # decline even when one cadence spans thousands of t_dec.
                bounds=[math.log(t) for t in [lo,hi]+[
                    t for t in (s.t_j_s,s.t_nr_s) if lo<t<hi]]
                integral=th.gauss_integral(lambda logt:
                    np.exp(logt)*s.lightcurve(np.exp(logt))**b,bounds,128)
                out.append(q*integral/cadence)
            return np.array(out)
        numeric=th.gauss_integral(density,th.q_bounds(s,cadence,1,"random_phase"),128)
        phase=th.phase_averaged_moment(alpha,cadence,1)
        self.assertRelative(phase,numeric)
        continuous=th.angular_moment(alpha)
        self.assertLess(phase,continuous)
        small=th.phase_averaged_moment(alpha,1e-4,1)
        self.assertLess(small,continuous)
        self.assertRelative(small,continuous,1e-5)

    def test_luminosity_first_and_distance_first_integrals(self):
        # Direct logarithmic integrations with analytically bounded remote tails.
        for alpha in [-2.4, -2, -1.75, -1.25, -1.05]:
            b, k, g, span = -alpha-1, alpha+2.5, .002, 80
            lum = (th.gauss_integral(lambda x:np.exp(k*x), [-span,0], 128)
                   +th.gauss_integral(lambda x:np.exp(-b*x), [0,span], 128)
                   +math.exp(-k*span)/k+math.exp(-b*span)/b)*g**b
            distance = 3*g**b/b*(th.gauss_integral(
                lambda x:np.exp((3-2*b)*x), [-span,0], 128)
                +math.exp(-(3-2*b)*span)/(3-2*b))
            self.assertRelative(lum, distance)
            self.assertRelative(lum, 3*g**b/(b*(3-2*b)))

    def test_exact_scalings_and_pivot_invariance(self):
        for alpha in [-2.4, -2, -1.25]:
            base = dict(A=3.0, alpha=alpha, D_euc_gpc=5.0, F_lim_Jy=3e-5,
                        J=th.angular_moment(alpha))
            r = th.rate(**base)
            self.assertRelative(th.rate(**dict(base, A=6)), 2*r)
            self.assertRelative(th.rate(**dict(base, F_lim_Jy=3e-4)), r*10**(alpha+1))
            self.assertRelative(th.rate(**dict(base, D_euc_gpc=10)), r*2**(2*alpha+5))
            self.assertRelative(th.rate(**dict(base, A=3*10**(alpha+1)), L_ref=1e33), r)

    def test_cdfs_medians_and_known_benchmarks(self):
        for alpha in [-2.4, -2, -1.75, -1.25, -1.05]:
            self.assertAlmostEqual(th.distance_cdf(th.distance_median(alpha), alpha), .5)
            self.assertAlmostEqual(th.conditional_luminosity_cdf(1, alpha), 2*(-alpha-1)/3)
            for prescription in ["continuous", "legacy", "peak_step", "random_phase"]:
                kw = dict(prescription=prescription,cadence_s=2*th.DAY_S,i_det=2)
                qm = th.angular_median(alpha, **kw)
                lm = th.luminosity_median(alpha, **kw)
                self.assertAlmostEqual(th.angular_cdf(qm,alpha,**kw), .5, places=8)
                self.assertAlmostEqual(th.luminosity_cdf(lm,alpha,**kw), .5, places=8)
        self.assertRelative(th.angular_moment(-2), .5489051724484098)
        self.assertRelative(th.luminosity_median(-2), .5778665464842161)

    def test_global_phase_luminosity_cdf_independent_two_dimensional_integral(self):
        s = th.ModelShape()
        cadence,count=2*th.DAY_S,2
        for alpha,ell in [(-2,1e4),(-2,1e6),(-1.25,1e9)]:
            b=-alpha-1
            threshold=s.fading_time(1/ell)
            def numerator(q_values):
                out=[]
                for q in q_values:
                    lo=float(s.peak_time(q))+(count-1)*cadence
                    hi=lo+cadence
                    bounds=[lo,hi]+[t for t in (s.t_j_s,s.t_nr_s,threshold) if lo<t<hi]
                    val=th.gauss_integral(lambda t:s.lightcurve(t)**b*
                        th.conditional_luminosity_cdf(ell*s.lightcurve(t),alpha),bounds,96)
                    out.append(q*val/cadence)
                return np.array(out)
            bounds=th.q_bounds(s,cadence,count,"random_phase",extra_times=[
                threshold-(count-1)*cadence,threshold-count*cadence])
            independent=th.gauss_integral(numerator,bounds,96)/th.phase_averaged_moment(alpha,cadence,count)
            self.assertRelative(th.luminosity_cdf(ell,alpha,cadence_s=cadence,
                i_det=count,prescription="random_phase"),independent)

    def test_extreme_convergent_medians_remain_available_in_logarithms(self):
        s=th.ModelShape()
        alpha=-1.001
        b,k=-alpha-1,alpha+2.5
        for prescription in ("continuous","random_phase"):
            kw=dict(prescription=prescription,cadence_s=2*th.DAY_S,i_det=2)
            logmed=th.log10_luminosity_median(alpha,**kw)
            self.assertTrue(math.isfinite(logmed))
            self.assertGreater(logmed,math.log10(np.finfo(float).max))
            self.assertEqual(th.luminosity_median(alpha,**kw),math.inf)
            # Evaluate the exact global bright-tail survival in log space;
            # no enormous luminosity or overflowing power is constructed.
            logtail=(math.log(2*k/3)+math.log(.5*s.q_nr**2)
                     -math.log(th.angular_moment(alpha,**kw))
                     -b*math.log(10)*logmed)
            self.assertAlmostEqual(logtail,math.log(.5),places=12)
        alpha=-2.4999
        logmed=th.log10_luminosity_median(alpha)
        self.assertTrue(math.isfinite(logmed))
        self.assertEqual(th.luminosity_median(alpha),0.)
        logcdf=(math.log(2*(-alpha-1)/3)+(alpha+2.5)*math.log(10)*logmed
                +math.log(th.angular_moment(-2.5))-math.log(th.angular_moment(alpha)))
        self.assertAlmostEqual(logcdf,math.log(.5),places=12)

    def test_finite_bounds_and_exact_omitted_tail_accounting(self):
        for alpha in [-2.4, -2, -1.25, -1.05]:
            for prescription in ["continuous", "legacy"]:
                kw = dict(prescription=prescription,cadence_s=2*th.DAY_S,i_det=2)
                missing = th.omitted_tail_fractions(1e-10,1e20,alpha,**kw)
                kept = th.finite_rate_ratio(1e-10,1e20,alpha,**kw)
                self.assertAlmostEqual(kept+missing["total"], 1, places=8)

    def test_convergence_boundaries_are_excluded(self):
        for alpha in [-3,-2.5,-1,0]:
            with self.assertRaises(ValueError):
                th.rate(1,alpha,5,3e-5,1)
        for alpha, low1, low2, high1, high2 in [(-2.5,1e-4,1e-8,1e12,1e12),
                (-1,1e-8,1e-8,1e14,1e18),(-3,1e-4,1e-8,1e12,1e12),
                (-.5,1e-8,1e-8,1e14,1e18)]:
            a = th.finite_luminosity_weight(low1,high1,1,alpha)
            b = th.finite_luminosity_weight(low2,high2,1,alpha)
            self.assertGreater(b,a)

    def test_corner_approximation_identity(self):
        # Independently integrate the luminosity-dependent thesis corner box.
        s = th.ModelShape()
        alpha = -2
        for cadence,prescription in [(None,"continuous"),(2*th.DAY_S,"legacy")]:
            info = th.corner_ratios(alpha,cadence_s=cadence,prescription=prescription)
            low_wall = info["L_wall_over_Ldec"]
            high_wall = 1/float(s.peak_shape(s.q_nr))
            qcap = info["q_corner"]
            def density(logL):
                L = np.exp(logL)
                times = np.array([s.fading_time(1/x) for x in L])
                angles = 1+np.where(times<s.t_j_s,(times/s.t_j_s)**(3/8),(times/s.t_j_s)**.5)
                angles = np.clip(angles,qcap,s.q_nr)
                box = .5*angles**2
                return L**(alpha+1)*box
            middle = th.gauss_integral(density,
                [math.log(low_wall), math.log(max(low_wall,1/float(s.lightcurve(s.t_j_s)))),
                 math.log(high_wall)],128)
            faint = .5*qcap**2*low_wall**(alpha+1)/(alpha+2.5)
            bright = .5*s.q_nr**2*high_wall**(alpha+1)/(-alpha-1)
            exact = 3*th.angular_moment(alpha,cadence_s=cadence,prescription=prescription)
            self.assertRelative((middle+faint+bright)/exact,
                                info["integrated_thesis_corner_over_exact"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
