from numbers import Integral as N
from numbers import Real as R

import numpy as np


def annuity_matching_T(
    p1: R, s1: R, r1: R, r2: R, tau: R, delta: R, gamma: R
) -> R:
    s1 = s1
    r1 = r1 / 12
    a1 = s1 * (1 - tau) * delta / 12

    s2 = s1 * (1 + gamma)
    r2 = r2 / 12
    a2 = s2 * (1 - tau) * delta / 12

    g = np.log(1 / (1 + r2)) / np.log(1 / (1 + r1))
    p2 = a2 / r2 * (1 - (1 - r1 * p1 / a1) ** g)
    return p2


def amortized_matching_T(
    p1: R, s1: R, r1: R, r2: R, tau: R, delta: R, gamma: R
) -> R:
    s1 = s1

    s1 = s1
    r1 = r1 / 12
    a1 = s1 * (1 - tau) * delta / 12

    s2 = s1 * (1 + gamma)
    r2 = r2 / 12
    a2 = s2 * (1 - tau) * delta / 12

    TP1 = p1 / a1
    p2 = 1 / (1 + r2 * TP1) * (a2 / a1 * (p1 * (1 + r1 * TP1) - a1 * r1 * TP1 * (TP1 - 1) / 2) + a2 * r2 * TP1 * (TP1 - 1) / 2)
    return p2
