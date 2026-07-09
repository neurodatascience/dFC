"""
API Conformance Checks for dFC Methods

Structural contract verification that runs before scientific test cases.
Each sub-check is independent: a failure in one does not prevent the others
from running, and every sub-check result is recorded separately.

The six sub-checks, in order:
  1. registry_instantiation  — class resolves from MEASURE_NAME via _build_measure_registry
  2. estimate_FCS_returns_self — group-level call returns the same object, no exception
  3. estimate_dFC_returns_DFC  — per-subject call returns a pydfc.dfc.DFC instance
  4. dfc_mat_shape             — get_dFC_mat() yields a 3-D [n_time, R, R] array
  5. symmetry                 — each sampled FC matrix equals its own transpose
  6. finite_values            — no NaN or Inf in any entry of get_dFC_mat()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SubCheckResult:
    """Result of a single API conformance sub-check."""

    name: str
    passed: bool
    error: str = ""  # empty string when passed


# ──────────────────────────────────────────────────────────────────────────────
# Conformance checker
# ──────────────────────────────────────────────────────────────────────────────


class APIConformanceCheck:
    """
    Run six structural sub-checks on a dFC method class identified by its
    MEASURE_NAME string.

    Designed to be called once per method before any scientific test case.
    A broad set of default parameters is supplied at instantiation time so
    that most methods can be initialised without extra configuration; methods
    that require parameters not in the default set will record a failure in
    sub-check 1 and skip the remaining checks.

    Parameters
    ----------
    n_regions : int
        Number of brain regions in the synthetic TIME_SERIES objects.
    n_timepoints : int
        Number of time points per subject.
    n_subjects : int
        Number of subjects in the group TIME_SERIES used for estimate_FCS.
    Fs : float
        Sampling frequency (Hz).
    """

    # Broad parameter set that covers the most common method requirements.
    # Methods ignore keys they do not recognise (all use params.get() or
    # key-in-dict lookup), so extra keys are always safe.
    _DEFAULT_PARAMS: dict = dict(
        # SlidingWindow-specific
        sw_method="pear_corr",
        n_overlap=0,
        tapered_window=False,
        # window size used by windowed methods
        W=10,
        # state-based methods
        n_states=3,
        n_subj_clstrs=2,
        hmm_iter=2,
        # DiscreteHMM / SlidingWindowClustr
        clstr_base_measure="SlidingWindow",
        clstr_distance="manhattan",
        dhmm_obs_state_ratio=2,
        # TimeFreq
        TF_method="WTC",
        # shared preprocessing
        normalization=True,
        num_select_nodes=None,
        num_time_point=None,
        Fs_ratio=None,
        noise_ratio=None,
        num_realization=None,
        session=None,
        # MTD and exponential/min_periods methods
        sigma_s=2.0,
        min_periods=5,
        half_life=10,
    )

    def __init__(
        self,
        n_regions: int = 12,
        n_timepoints: int = 80,
        n_subjects: int = 3,
        Fs: float = 1.0,
    ) -> None:
        self.n_regions = n_regions
        self.n_timepoints = n_timepoints
        self.n_subjects = n_subjects
        self.Fs = Fs

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_time_series(self):
        """
        Return (group_ts, subj_ts): a multi-subject TIME_SERIES for
        estimate_FCS and a single-subject one for estimate_dFC.
        """
        from pydfc.time_series import TIME_SERIES

        rng = np.random.RandomState(0)
        locs = np.zeros((self.n_regions, 3), dtype=float)
        node_labels = [f"r{i:02d}" for i in range(self.n_regions)]

        def _ts(subj_id: str):
            return TIME_SERIES(
                data=rng.randn(self.n_regions, self.n_timepoints),
                subj_id=subj_id,
                Fs=self.Fs,
                locs=locs,
                node_labels=node_labels,
            )

        group_ts = _ts("sub_000")
        for si in range(1, self.n_subjects):
            group_ts.append_ts(
                new_time_series=rng.randn(self.n_regions, self.n_timepoints),
                subj_id=f"sub_{si:03d}",
            )

        subj_ts = _ts("sub_000")
        return group_ts, subj_ts

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, measure_name: str) -> List[SubCheckResult]:
        """
        Execute all six sub-checks for *measure_name* and return one
        SubCheckResult per check.  Checks are independent: an exception in
        one is caught and recorded; subsequent checks that depend on an
        earlier result are marked as skipped with an explanatory message.

        Parameters
        ----------
        measure_name : str
            The MEASURE_NAME class attribute of the target dFC method.

        Returns
        -------
        List[SubCheckResult]
            Always exactly six items, one per sub-check.
        """
        results: List[SubCheckResult] = []
        SKIP = "Skipped: earlier required sub-check failed"

        # ── sub-check 1: registry instantiation ───────────────────────────────
        method = None
        try:
            from pydfc.multi_analysis_utils import create_measure_obj

            (method,) = create_measure_obj([measure_name], **self._DEFAULT_PARAMS)
            results.append(SubCheckResult("registry_instantiation", True))
        except Exception as exc:
            results.append(SubCheckResult("registry_instantiation", False, str(exc)))
            for name in (
                "estimate_FCS_returns_self",
                "estimate_dFC_returns_DFC",
                "dfc_mat_shape",
                "symmetry",
                "finite_values",
            ):
                results.append(SubCheckResult(name, False, SKIP))
            return results

        # ── build TIME_SERIES (required for checks 2-6) ───────────────────────
        try:
            group_ts, subj_ts = self._make_time_series()
        except Exception as exc:
            msg = f"TIME_SERIES construction failed: {exc}"
            for name in (
                "estimate_FCS_returns_self",
                "estimate_dFC_returns_DFC",
                "dfc_mat_shape",
                "symmetry",
                "finite_values",
            ):
                results.append(SubCheckResult(name, False, msg))
            return results

        # ── sub-check 2: estimate_FCS returns self ────────────────────────────
        try:
            ret = method.estimate_FCS(time_series=group_ts)
            if ret is not method:
                raise AssertionError(
                    f"estimate_FCS returned {type(ret).__name__!r}, expected self"
                )
            results.append(SubCheckResult("estimate_FCS_returns_self", True))
        except Exception as exc:
            results.append(SubCheckResult("estimate_FCS_returns_self", False, str(exc)))
            # estimate_dFC may still work independently; do not skip

        # ── sub-check 3: estimate_dFC returns a DFC object ────────────────────
        dfc_obj = None
        try:
            from pydfc.dfc import DFC

            dfc_obj = method.estimate_dFC(time_series=subj_ts)
            if not isinstance(dfc_obj, DFC):
                raise AssertionError(
                    f"estimate_dFC returned {type(dfc_obj).__name__!r}, expected DFC"
                )
            results.append(SubCheckResult("estimate_dFC_returns_DFC", True))
        except Exception as exc:
            results.append(SubCheckResult("estimate_dFC_returns_DFC", False, str(exc)))
            for name in ("dfc_mat_shape", "symmetry", "finite_values"):
                results.append(SubCheckResult(name, False, SKIP))
            return results

        # ── sub-check 4: get_dFC_mat shape ────────────────────────────────────
        mat = None
        try:
            mat = dfc_obj.get_dFC_mat()
            if mat.ndim != 3:
                raise AssertionError(f"Expected 3-D array, got shape {mat.shape}")
            if mat.shape[1] != self.n_regions or mat.shape[2] != self.n_regions:
                raise AssertionError(
                    f"Expected [..., {self.n_regions}, {self.n_regions}], "
                    f"got {mat.shape}"
                )
            results.append(SubCheckResult("dfc_mat_shape", True))
        except Exception as exc:
            results.append(SubCheckResult("dfc_mat_shape", False, str(exc)))

        if mat is None:
            for name in ("symmetry", "finite_values"):
                results.append(SubCheckResult(name, False, SKIP))
            return results

        # ── sub-check 5: symmetry ─────────────────────────────────────────────
        try:
            n_check = min(mat.shape[0], 10)
            for t in range(n_check):
                if not np.allclose(mat[t], mat[t].T, atol=1e-10):
                    raise AssertionError(f"Frame {t} is not symmetric")
            results.append(SubCheckResult("symmetry", True))
        except Exception as exc:
            results.append(SubCheckResult("symmetry", False, str(exc)))

        # ── sub-check 6: finite values ────────────────────────────────────────
        try:
            n_nonfinite = int(np.sum(~np.isfinite(mat)))
            if n_nonfinite > 0:
                raise AssertionError(f"Output contains {n_nonfinite} non-finite value(s)")
            results.append(SubCheckResult("finite_values", True))
        except Exception as exc:
            results.append(SubCheckResult("finite_values", False, str(exc)))

        return results
