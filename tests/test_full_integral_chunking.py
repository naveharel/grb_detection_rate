"""Regression tests for the memory-bounded q-chunking of
rate_log10_full_integral.

The full-integral rate builds (N_q, *grid) tensors; on large strategy grids
(e.g. the optimizer's 280x320 refine pass) with s_rise > 0 the unchunked form
peaked at ~5.5 GiB — beyond Pyodide's WASM32 ~4 GiB address space
(MemoryError in the browser).  The integral is now accumulated over q-chunks
of at most _FULL_INTEGRAL_CHUNK_ELEMS elements per intermediate; adjacent
chunks share their boundary q-point, so the chunked trapezoid equals the
single-pass trapezoid up to float summation order.

Run with::

    .venv/Scripts/python -m pytest tests/test_full_integral_chunking.py -v
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from grb_detect import detection_rate
from grb_detect.constants import DAY_S
from grb_detect.detection_rate import DetectionRateModel
from grb_detect.params import AfterglowPhysicalParams, SurveyInstrumentParams


def _make_default_model() -> DetectionRateModel:
    return DetectionRateModel(AfterglowPhysicalParams(), SurveyInstrumentParams())


@pytest.fixture(scope="module")
def model() -> DetectionRateModel:
    return _make_default_model()


@pytest.fixture(scope="module")
def small_grid() -> tuple[np.ndarray, np.ndarray]:
    N = np.logspace(0, 4, 12)
    t = np.logspace(np.log10(60), np.log10(86400 * 30), 10)
    return np.meshgrid(N, t, indexing="ij")  # 12 x 10 = 120 cells


# --------------------------------------------------------------------------- #
# C.1 - chunked accumulation matches the single-chunk integral                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "s_fade,s_rise", [(0.0, 0.0), (0.5, 0.0), (0.0, 1.5), (0.5, 1.5)]
)
@pytest.mark.parametrize("N_q", [97, 100, 500])
def test_C01_chunked_matches_single_chunk(
    model, small_grid, monkeypatch, s_fade, s_rise, N_q
):
    """All filter branches, ragged and exact chunk splits."""
    N, T = small_grid
    kw = dict(
        q_min=1.05,
        D_min_cm=0.1 * model.phys.D_euc_cm,
        s_fade=s_fade,
        s_rise=s_rise,
        s_mode="discrete",
        N_q=N_q,
    )
    R_single = model.rate_log10_full_integral(3, N, T, **kw)  # default budget
    monkeypatch.setattr(detection_rate, "_FULL_INTEGRAL_CHUNK_ELEMS", 500)
    R_chunked = model.rate_log10_full_integral(3, N, T, **kw)  # 4-point chunks
    np.testing.assert_allclose(R_chunked, R_single, rtol=1e-12, equal_nan=True)


def test_C02_minimum_two_point_chunks(model, small_grid, monkeypatch):
    """Budget below grid size hits the max(2, ...) floor: one q-interval per
    chunk — the finest possible split still covers every interval once."""
    N, T = small_grid
    R_single = model.rate_log10_full_integral(3, N, T, s_rise=1.0, N_q=64)
    monkeypatch.setattr(detection_rate, "_FULL_INTEGRAL_CHUNK_ELEMS", 1)
    R_chunked = model.rate_log10_full_integral(3, N, T, s_rise=1.0, N_q=64)
    np.testing.assert_allclose(R_chunked, R_single, rtol=1e-12, equal_nan=True)


def test_C03_one_element_grid_chunked(model, monkeypatch):
    """1-element strategy arrays (point-evaluation callers) survive chunking."""
    N, T = np.array([30.0]), np.array([2.0 * DAY_S])
    R_single = model.rate_log10_full_integral(3, N, T, s_fade=0.4, s_rise=0.8)
    monkeypatch.setattr(detection_rate, "_FULL_INTEGRAL_CHUNK_ELEMS", 16)
    R_chunked = model.rate_log10_full_integral(3, N, T, s_fade=0.4, s_rise=0.8)
    np.testing.assert_allclose(R_chunked, R_single, rtol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# C.4 - large-grid s_rise > 0 evaluation stays memory-bounded                  #
# --------------------------------------------------------------------------- #
def test_C04_large_grid_srise_memory_bounded(model):
    """Pre-chunking, this (160, 140) grid peaked at ~1.3 GiB (and the real
    280x320 refine grid at ~5.5 GiB, the WASM32 OOM).  The chunked peak is
    grid-size-independent (~250 MiB); numpy registers its buffers with
    tracemalloc, so the measured peak is honest."""
    N, T = np.meshgrid(np.logspace(0, 3.5, 140), np.logspace(1, 7.5, 160))
    tracemalloc.start()
    try:
        R = model.rate_log10_full_integral(
            3, N, T, s_fade=0.5, s_rise=1.5, N_q=500
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert R.shape == (160, 140)
    assert np.any(np.isfinite(R))
    assert peak < 600 * 1024**2, (
        f"peak {peak / 2**20:.0f} MiB exceeds the chunked-memory budget"
    )
