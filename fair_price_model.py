from numbers import Integral as N
from numbers import Real as R

import numpy as np

from loan_model import get_loan_term, get_payment, get_rate


def foo(p1, n1, r1, r2, rf, t2, t1):
    a1 = get_payment(p1, r1, n1)
    # a1 = p1*get_rate(r1, n1)

    r1 = r1 / 12
    r2 = r2 / 12

    # def get_n(p, a, r):
    #    return np.log(1-r*p/a)/np.log(1/(1+r))

    n1 = n1 * 12
    n2 = get_loan_term(p1, a1, r2)  # since p1/a1 = p2/a2 this is OK

    print(f"Loan duration (t1): {n1/12:.2f} Y")
    print(f"Loan duration (t2): {n2/12:.2f} Y")

    k = r1 / (1 - (1 / (1 + r1)) ** n1) * p1 * n1

    p2 = (1 - (1 / (1 + r2)) ** n2) / n2 / r2 * k * np.exp(rf * (t2 - t1))
    print(f"Fair price: {p2:,.0f}")
    print(f"Relative diff: {(p2-p1)/p1:.4%}")


def equivalent_principal_matching_T(
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
