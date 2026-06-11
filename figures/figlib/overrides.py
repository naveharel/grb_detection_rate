"""Copy-on-write "model surgery": vary quantities the public API does not expose.

The public factory ``make_rate_model`` exposes survey + physics parameters, but not
the *derived* internal scales (deceleration/jet/non-relativistic angles, and the flux
normalizations ``F_dec``/``F_j``/``F_nr``). Figures frequently need to sweep those.

This module varies them WITHOUT modifying the read-only engine, by surgically
replacing fields on a *copy* of a built model.

Why a copy (never in place)
---------------------------
``make_rate_model`` is ``lru_cache``d: two identical builds return the *same* object.
Mutating ``model._derived`` in place would corrupt every other holder of that cached
instance. ``DetectionRateModel`` is a plain class exposing a read-only ``derived``
property over ``self._derived``; ``DerivedAfterglowScales`` is a frozen dataclass. So
the safe pattern is ``copy.copy(model)`` (new object, shared frozen phys/instrument/
micro/pls) followed by binding a freshly ``dataclasses.replace``-d ``_derived``.

Contract for new override helpers
---------------------------------
1. Operate on a copy; never mutate a cached instance.
2. When a field is physically coupled to others, scale/replace them *together*
   (see ``set_F_dec`` — F_j and F_nr are tied to F_dec).
3. Document the physics invariant the override relies on, and prefer rebuilding via
   ``make_rate_model`` for anything the public API already exposes (physics params);
   reserve surgery for the derived/internal scales it hides.
"""

from __future__ import annotations

import copy
import dataclasses


def copy_model(model):
    """Return a shallow copy of ``model``.

    The copy is a distinct object whose ``phys``/``instrument``/``micro``/``pls`` and
    ``_derived`` are shared (all frozen / safe to share) with the original. Reassigning
    ``_derived`` on the copy does not affect the original or any lru-cached instance.
    """
    return copy.copy(model)


def override_derived(model, **field_overrides):
    """Return a copy of ``model`` with ``DerivedAfterglowScales`` fields replaced.

    Generic surgery for any derived scale: ``t_dec_s, t_j_s, q_dec, q_j, q_nr,
    F_dec_Jy, F_j_Jy, F_nr_Jy``. The caller is responsible for keeping physically
    coupled fields consistent (use the specialized helpers below where they exist).
    """
    m = copy.copy(model)
    m._derived = dataclasses.replace(model.derived, **field_overrides)
    return m


def set_F_dec(model, F_dec_Jy: float):
    """Return a copy of ``model`` with the flux normalization set to ``F_dec_Jy`` [Jy].

    This is the physically-faithful "override only the flux normalization" operation.

    Invariant (verified against the engine): ``F_j`` and ``F_nr`` are ``F_dec`` times
    pure functions of ``q_dec``/``q_nr`` (independent of ``F_dec``), and flux enters the
    q-computation only through ``q_Euc()``/``D_dec()`` as the ratio ``F_lim/F_dec`` and
    the comparison ``F_lim < F_j``. Therefore scaling ``F_dec_Jy``, ``F_j_Jy`` and
    ``F_nr_Jy`` by the SAME factor is exactly equivalent to physically changing
    ``F_dec`` while leaving every angular/temporal exponent (and ``q_dec/q_j/q_nr/
    t_dec``) untouched.
    """
    d = model.derived
    F_cur = float(d.F_dec_Jy)
    if not (F_cur > 0.0):
        raise ValueError(f"current F_dec_Jy must be positive to rescale (got {F_cur!r})")
    k = float(F_dec_Jy) / F_cur
    return override_derived(
        model,
        F_dec_Jy=d.F_dec_Jy * k,
        F_j_Jy=d.F_j_Jy * k,
        F_nr_Jy=d.F_nr_Jy * k,
    )


def scale_F_dec(model, factor: float):
    """Return a copy of ``model`` with the flux normalization multiplied by ``factor``."""
    return set_F_dec(model, model.derived.F_dec_Jy * float(factor))
