"""Psychometric meta-analysis functions (Hunter-Schmidt bare-bones + sensitivity checks)."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BareBones:
    k: int
    total_n: float
    mean_r: float
    sd_r: float
    var_e: float
    sd_res: float
    se_mean: float
    ci_lo: float
    ci_hi: float
    cv_lo: float
    cv_hi: float

    def corrected(self, rxx_p: float = 0.80, rxx_c: float = 0.85) -> tuple[float, float, float]:
        """Correct mean and credibility bounds for measurement error in both variables."""
        a = np.sqrt(rxx_p * rxx_c)
        return self.mean_r / a, self.cv_lo / a, self.cv_hi / a


def bare_bones(r: np.ndarray, n: np.ndarray) -> BareBones:
    """N-weighted Hunter-Schmidt bare-bones meta-analysis."""
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    k = len(r)
    w = n
    mean_r = np.sum(w * r) / np.sum(w)
    var_r = np.sum(w * (r - mean_r) ** 2) / np.sum(w)
    var_e = np.sum(w * ((1 - mean_r**2) ** 2 / (n - 1))) / np.sum(w)
    var_res = max(var_r - var_e, 0.0)
    sd_res = np.sqrt(var_res)
    se_mean = np.sqrt(var_r / k)
    z = stats.norm.ppf(0.975)
    z80 = stats.norm.ppf(0.90)
    return BareBones(
        k=k,
        total_n=float(np.sum(n)),
        mean_r=float(mean_r),
        sd_r=float(np.sqrt(var_r)),
        var_e=float(var_e),
        sd_res=float(sd_res),
        se_mean=float(se_mean),
        ci_lo=float(mean_r - z * se_mean),
        ci_hi=float(mean_r + z * se_mean),
        cv_lo=float(mean_r - z80 * sd_res),
        cv_hi=float(mean_r + z80 * sd_res),
    )


def subgroup_contrast(a: BareBones, b: BareBones) -> tuple[float, float, float]:
    """Difference in mean r between two subgroups with z test (independent samples)."""
    diff = a.mean_r - b.mean_r
    se = np.sqrt(a.se_mean**2 + b.se_mean**2)
    z = diff / se
    p = 2 * stats.norm.sf(abs(z))
    return diff, z, p


def dl_random_effects(r: np.ndarray, n: np.ndarray) -> tuple[float, float, float, float]:
    """DerSimonian-Laird random-effects on Fisher-z scale (sensitivity analysis).

    Returns mean r (back-transformed), 95% CI bounds, and tau (Fisher-z scale).
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.arctanh(np.clip(r, -0.999, 0.999))
    v = 1.0 / (n - 3)
    w = 1.0 / v
    ybar = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - ybar) ** 2)
    df = len(y) - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max((q - df) / c, 0.0)
    w_star = 1.0 / (v + tau2)
    mu = np.sum(w_star * y) / np.sum(w_star)
    se = np.sqrt(1.0 / np.sum(w_star))
    zc = stats.norm.ppf(0.975)
    return (
        float(np.tanh(mu)),
        float(np.tanh(mu - zc * se)),
        float(np.tanh(mu + zc * se)),
        float(np.sqrt(tau2)),
    )
