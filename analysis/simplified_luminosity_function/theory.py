"""Scale-free detection-rate theory; no intrinsic luminosity PDF is assumed.

Luminosity means L_nu(t_dec), in erg/s/Hz.  The intensity is
d rho/dL = A/L_ref (L/L_ref)**alpha, with A in Gpc^-3 yr^-1.
This analysis does not modify or replace the application's physics engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

DAY_S = 86400.0
GPC_CM = 3.0856775814913673e27
L_REF = 1e32


def power_integral(lo, hi, exponent):
    """Integral x**exponent dx, including the logarithmic limit."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
    hi = np.maximum(hi, lo)
    e = exponent + 1.0
    if abs(e) < 1e-12:
        return np.log(hi / lo)
    return lo**e * np.expm1(e * np.log(hi / lo)) / e


@lru_cache(maxsize=12)
def gauss_rule(n):
    return np.polynomial.legendre.leggauss(n)


def gauss_integral(fn, bounds, n=128):
    """Independent segmented Gauss-Legendre quadrature."""
    x, w = gauss_rule(n)
    bounds = sorted(set(float(t) for t in bounds))
    return sum((hi-lo)/2 * float(np.dot(w, fn((hi+lo)/2+(hi-lo)/2*x)))
               for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo)


def convergent_alpha(alpha):
    if not -2.5 < alpha < -1.0:
        raise ValueError("Finite detected rate requires -5/2 < alpha < -1")
    return -alpha - 1.0


@dataclass(frozen=True)
class ModelShape:
    p: float = 2.5
    theta_j: float = 0.1
    gamma0: float = 10**2.5
    t_dec_s: float = 19.4

    @property
    def q_dec(self):
        return 1.0 + 1.0/(self.gamma0*self.theta_j)

    @property
    def q_nr(self):
        return math.sqrt(2)/self.theta_j

    @property
    def t_j_s(self):
        return self.t_dec_s*(self.gamma0*self.theta_j)**(8/3)

    @property
    def t_nr_s(self):
        return self.t_j_s*(self.q_nr-1)**2

    @property
    def a_II(self):
        return 3*(self.p-1)/4

    @property
    def a_IV(self):
        return (15*self.p-21)/10

    def lightcurve(self, t, late_time=True):
        """Continuous thesis II/III/IV decline, normalized at t_dec.

        Values before t_dec are clamped to the peak; pre-deceleration rise is
        not used by the post-peak selections in this analysis.
        """
        t = np.maximum(np.asarray(t, float), self.t_dec_s)
        gj = (self.t_j_s/self.t_dec_s)**(-self.a_II)
        gnr = gj*(self.t_nr_s/self.t_j_s)**(-self.p)
        ans = np.where(t < self.t_j_s, (t/self.t_dec_s)**(-self.a_II),
                       gj*(t/self.t_j_s)**(-self.p))
        if late_time:
            ans = np.where(t > self.t_nr_s,
                           gnr*(t/self.t_nr_s)**(-self.a_IV), ans)
        return ans

    def peak_time(self, q):
        q = np.asarray(q, float)
        u = np.maximum(q-1, 0)
        return np.maximum(self.t_dec_s,
                          self.t_j_s*np.where(q < 2, u**(8/3), u**2))

    def peak_shape(self, q):
        return self.lightcurve(self.peak_time(q))

    def inverse_peak_time(self, t):
        return 1+(t/self.t_j_s)**(3/8 if t < self.t_j_s else .5)

    def fading_time(self, g, late_time=True):
        """Time at which the declining light curve equals 0 < g <= 1."""
        g = float(g)
        if g >= 1:
            return self.t_dec_s
        if g <= 0:
            return math.inf
        gj = float(self.lightcurve(self.t_j_s))
        gn = float(self.lightcurve(self.t_nr_s))
        if g >= gj:
            return self.t_dec_s*g**(-1/self.a_II)
        if not late_time or g >= gn:
            return self.t_j_s*(g/gj)**(-1/self.p)
        return self.t_nr_s*(g/gn)**(-1/self.a_IV)

    def time_moment(self, lo, hi, exponent, late_time=True):
        """Exact integral of lightcurve(t)**exponent dt, including log cases."""
        lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
        hi = np.maximum(hi, lo)
        ans = np.maximum(np.minimum(hi, self.t_dec_s)-lo, 0)
        edges = [self.t_dec_s, self.t_j_s]
        if late_time:
            edges.append(self.t_nr_s)
        edges.append(math.inf)
        slopes = [self.a_II, self.p] + ([self.a_IV] if late_time else [])
        for t0, t1, a in zip(edges[:-1], edges[1:], slopes):
            lower = np.maximum(lo, t0)
            upper = np.maximum(lower, np.minimum(hi, t1))
            amp = float(self.lightcurve(t0, late_time))**exponent
            ans = ans + amp*t0*power_integral(lower/t0, upper/t0, -a*exponent)
        return ans


DEFAULT_SHAPE = ModelShape()


def selection_shape(q, cadence_s=None, i_det=2, prescription="continuous",
                    late_time=True, shape=DEFAULT_SHAPE):
    tp = shape.peak_time(q)
    if prescription == "continuous":
        times = tp
    elif prescription == "legacy":
        times = np.maximum(tp, i_det*cadence_s)
    elif prescription == "peak_step":
        times = tp+i_det*cadence_s
    else:
        raise ValueError("Use deterministic continuous, legacy, or peak_step")
    return shape.lightcurve(times, late_time)


def q_bounds(shape=DEFAULT_SHAPE, cadence_s=None, i_det=2,
             prescription="continuous", top=None, late_time=True, extra_times=()):
    top = shape.q_nr if top is None else float(np.clip(top, 0, shape.q_nr))
    bounds = [0, top, shape.q_dec, 2]
    times = list(extra_times)
    if prescription == "legacy":
        times += [i_det*cadence_s]
    elif prescription in ("peak_step", "random_phase"):
        offsets = [i_det*cadence_s]
        if prescription == "random_phase":
            offsets += [(i_det-1)*cadence_s]
        for offset in offsets:
            times += [shape.t_j_s-offset]
            if late_time:
                times += [shape.t_nr_s-offset]
    bounds += [shape.inverse_peak_time(t) for t in times if t > shape.t_dec_s]
    return sorted(set([0, top]+[v for v in bounds if 0 < v < top]))


def _angular_power(lo, hi, exponent):
    if hi <= lo:
        return 0.0
    return float(power_integral(lo, hi, 1-exponent)+power_integral(lo, hi, -exponent))


def analytic_angular_moment(alpha, cadence_s=None, i_det=2,
                            prescription="continuous", top=None,
                            late_time=True, shape=DEFAULT_SHAPE):
    """Exact J = integral q*g_selection**(-alpha-1) dq for legacy/continuous."""
    if prescription not in ("continuous", "legacy"):
        raise ValueError("Analytic q integral is available for continuous/legacy")
    b = -alpha-1
    top = shape.q_nr if top is None else float(np.clip(top, 0, shape.q_nr))
    T = shape.t_dec_s if prescription == "continuous" else max(shape.t_dec_s, i_det*cadence_s)
    qcap = min(shape.q_nr, shape.inverse_peak_time(T))
    gcap = float(shape.lightcurve(T, late_time))
    ans = .5*min(top, qcap)**2*gcap**b
    if top <= qcap:
        return ans
    gj = float(shape.lightcurve(shape.t_j_s))
    if qcap < 2:
        ans += gj**b*_angular_power(qcap-1, min(top, 2)-1, 2*(shape.p-1)*b)
    if top > 2:
        ans += gj**b*_angular_power(max(qcap, 2)-1, top-1, 2*shape.p*b)
    return ans


def moment_density(q, alpha, cadence_s=None, i_det=2,
                   prescription="continuous", late_time=True,
                   shape=DEFAULT_SHAPE):
    q = np.asarray(q, float)
    b = -alpha-1
    if prescription == "random_phase":
        lo = shape.peak_time(q)+(i_det-1)*cadence_s
        weight = shape.time_moment(lo, lo+cadence_s, b, late_time)/cadence_s
    else:
        weight = selection_shape(q, cadence_s, i_det, prescription, late_time, shape)**b
    return q*weight


def angular_moment(alpha, cadence_s=None, i_det=2, prescription="continuous",
                   method="analytic", n=128, top=None, late_time=True,
                   shape=DEFAULT_SHAPE):
    if method == "analytic" and prescription in ("continuous", "legacy"):
        return analytic_angular_moment(alpha, cadence_s, i_det, prescription,
                                       top, late_time, shape)
    bounds = q_bounds(shape, cadence_s, i_det, prescription, top, late_time)
    return gauss_integral(lambda q: moment_density(q, alpha, cadence_s, i_det,
                           prescription, late_time, shape), bounds, n)


def phase_averaged_moment(alpha, cadence_s, i_det=2, n=128,
                          late_time=True, shape=DEFAULT_SHAPE):
    return angular_moment(alpha, cadence_s, i_det, "random_phase", n=n,
                           late_time=late_time, shape=shape)


def rate(A, alpha, D_euc_gpc, F_lim_Jy, J, theta_j=.1, L_ref=L_REF, f_sky=1):
    b = convergent_alpha(alpha)
    Ldec = 4*math.pi*(D_euc_gpc*GPC_CM)**2*F_lim_Jy*1e-23
    volume = 4*math.pi/3*D_euc_gpc**3
    return f_sky*volume*theta_j**2*A*3/(b*(3-2*b))*(Ldec/L_ref)**(-b)*J


def distance_cdf(x, alpha):
    convergent_alpha(alpha)
    return np.clip(np.asarray(x, float), 0, 1)**(2*alpha+5)


def distance_median(alpha):
    convergent_alpha(alpha)
    return 2**(-1/(2*alpha+5))


def conditional_luminosity_cdf(x, alpha):
    b = convergent_alpha(alpha)
    k = 1.5-b
    x = np.maximum(np.asarray(x, float), 1e-300)
    return np.where(x <= 1, 2*b/3*np.minimum(x, 1)**k,
                    1-2*k/3*np.maximum(x, 1)**(-b))


def conditional_log_luminosity_density(x, alpha):
    b = convergent_alpha(alpha)
    k = 1.5-b
    x = np.maximum(np.asarray(x, float), 1e-300)
    return np.where(x <= 1, np.minimum(x, 1)**k,
                    np.maximum(x, 1)**(-b))/(1/k+1/b)


def angular_cdf(q, alpha, **selection):
    return angular_moment(alpha, top=q, **selection)/angular_moment(alpha, **selection)


def _bisect(fn, target, lo, hi, iterations=65):
    for _ in range(iterations):
        mid = (lo+hi)/2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def angular_median(alpha, **selection):
    shape = selection.get("shape", DEFAULT_SHAPE)
    return _bisect(lambda q: angular_cdf(q, alpha, **selection), .5, 0, shape.q_nr)


def luminosity_cdf(L_over_Ldec, alpha, cadence_s=None, i_det=2,
                   prescription="continuous", late_time=True,
                   shape=DEFAULT_SHAPE, n=128):
    """Global detected L CDF, using physical luminosity divided by L_dec.

    Phase-averaged selection integrates the luminosity CDF over time exactly;
    remaining angular integration is split at each light-curve transition.
    """
    ell = float(L_over_Ldec)
    if ell <= 0:
        return 0.0
    b = convergent_alpha(alpha)
    k = 1.5-b
    args = dict(cadence_s=cadence_s, i_det=i_det, prescription=prescription,
                late_time=late_time, shape=shape, n=n)
    J = angular_moment(alpha, **args)
    threshold_time = shape.fading_time(1/ell, late_time)
    if prescription == "random_phase":
        extra_times = [threshold_time-(i_det-1)*cadence_s,
                       threshold_time-i_det*cadence_s]
        def integrand(q):
            lo = shape.peak_time(q)+(i_det-1)*cadence_s
            hi = lo+cadence_s
            cross = np.clip(threshold_time, lo, hi) if ell >= 1 else lo
            high = (shape.time_moment(lo, cross, b, late_time)
                    -2*k/3*ell**(-b)*(cross-lo))
            low = 2*b/3*ell**k*shape.time_moment(cross, hi, 1.5, late_time)
            return q*(high+low)/cadence_s
    else:
        offset = i_det*cadence_s if prescription == "peak_step" else 0
        extra_times = [threshold_time-offset]
        def integrand(q):
            g = selection_shape(q, cadence_s, i_det, prescription, late_time, shape)
            return q*g**b*conditional_luminosity_cdf(ell*g, alpha)
    bounds = q_bounds(shape, cadence_s, i_det, prescription, None, late_time, extra_times)
    return float(np.clip(gauss_integral(integrand, bounds, n)/J, 0, 1))


def log10_luminosity_median(alpha, **selection):
    """Global log10(L_med/L_dec), including extreme convergent slopes.

    Below every wall or above every wall, the global CDF has an exact power
    tail. Solving that tail in logarithms avoids evaluating enormous or tiny
    luminosities. Only a median between the smallest and largest wall needs
    numerical CDF inversion. A finite logarithmic result can correspond to an
    ordinary luminosity outside the representable floating-point range.
    """
    b = convergent_alpha(alpha)
    k = 1.5-b
    shape = selection.get("shape", DEFAULT_SHAPE)
    prescription = selection.get("prescription", "continuous")
    cadence = selection.get("cadence_s")
    count = selection.get("i_det", 2)
    late_time = selection.get("late_time", True)
    Jb = angular_moment(alpha, **selection)
    J15 = angular_moment(-2.5, **selection)
    if prescription == "continuous":
        earliest, latest = shape.t_dec_s, shape.t_nr_s
    elif prescription == "legacy":
        earliest = max(shape.t_dec_s, count*cadence)
        latest = max(shape.t_nr_s, count*cadence)
    elif prescription == "peak_step":
        earliest = shape.t_dec_s+count*cadence
        latest = shape.t_nr_s+count*cadence
    elif prescription == "random_phase":
        earliest = shape.t_dec_s+(count-1)*cadence
        latest = shape.t_nr_s+count*cadence
    else:
        raise ValueError("Unknown selection prescription")
    min_log_wall = -math.log(float(shape.lightcurve(earliest, late_time)))
    max_log_wall = -math.log(float(shape.lightcurve(latest, late_time)))
    faint_log_median = (math.log(3/(4*b))+math.log(Jb)-math.log(J15))/k
    if faint_log_median <= min_log_wall:
        return faint_log_median/math.log(10)
    Q = .5*shape.q_nr**2
    bright_log_median = (math.log(4*k/3)+math.log(Q)-math.log(Jb))/b
    if bright_log_median >= max_log_wall:
        return bright_log_median/math.log(10)
    return _bisect(lambda logL: luminosity_cdf(10**logL, alpha, **selection),
        .5, min_log_wall/math.log(10), max_log_wall/math.log(10), iterations=65)


def luminosity_median(alpha, **selection):
    """Global L_med/L_dec; returns inf on float overflow, 0 on underflow.

    Use log10_luminosity_median to preserve extreme but mathematically finite
    quantiles near either convergence boundary.
    """
    logL = log10_luminosity_median(alpha, **selection)
    if logL > math.log10(np.finfo(float).max):
        return math.inf
    if logL < math.log10(np.nextafter(0.0, 1.0)):
        return 0.0
    return 10**logL


def corner_ratios(alpha, cadence_s=None, i_det=2,
                   prescription="continuous", late_time=True, shape=DEFAULT_SHAPE):
    """Distinguish one angular box from LF-integrated per-L thesis corners."""
    b = convergent_alpha(alpha)
    T = shape.t_dec_s if prescription == "continuous" else max(shape.t_dec_s, i_det*cadence_s)
    qcap = min(shape.q_nr, shape.inverse_peak_time(T))
    J = angular_moment(alpha, cadence_s, i_det, prescription,
                       late_time=late_time, shape=shape)
    core = .5*qcap**2*float(shape.lightcurve(T, late_time))**b/J
    return dict(q_corner=qcap, L_wall_over_Ldec=1/float(shape.lightcurve(T, late_time)),
                angular_box_over_exact=core,
                integrated_thesis_corner_over_exact=1-2*b/3*(1-core))


def finite_luminosity_weight(low, high, g, alpha):
    """Unnormalized integral L**alpha min((L*g)**1.5,1)dL; L is in L_dec units.

    Unlike the infinite result, this supports boundary/nonconvergent slopes.
    """
    g = np.asarray(g, float)
    cross = np.clip(1/g, low, high)
    return (g**1.5*power_integral(low, cross, alpha+1.5)
            +power_integral(cross, high, alpha))


def finite_rate_ratio(low, high, alpha, cadence_s=None, i_det=2,
                      prescription="continuous", shape=DEFAULT_SHAPE, n=128):
    b = convergent_alpha(alpha)
    bounds = q_bounds(shape, cadence_s, i_det, prescription)
    for ell in [low, high]:
        time = shape.fading_time(1/ell)
        if shape.t_dec_s < time < shape.t_nr_s:
            bounds.append(shape.inverse_peak_time(time))
    value = gauss_integral(lambda q:q*finite_luminosity_weight(low, high,
        selection_shape(q, cadence_s, i_det, prescription, shape=shape), alpha), bounds, n)
    J = angular_moment(alpha, cadence_s, i_det, prescription, shape=shape)
    return value/(3/(b*(3-2*b))*J)


def omitted_tail_fractions(low, high, alpha, **selection):
    """Exact missing tails when low is below every wall and high above every wall."""
    b = convergent_alpha(alpha)
    k = 1.5-b
    shape = selection.get("shape", DEFAULT_SHAPE)
    Jb = angular_moment(alpha, **selection)
    J15 = angular_moment(-2.5, **selection)
    faint = 2*b/3*low**k*J15/Jb
    bright = 2*k/3*high**(-b)*(.5*shape.q_nr**2)/Jb
    return dict(faint=faint, bright=bright, total=faint+bright)
