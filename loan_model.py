from numbers import Integral as N
from numbers import Real as R

import numpy as np


def get_rate(r: R, t: R) -> R:
    """
    Calculates monthly payment of an amortized loan with annual interest rate
    `r`, in decimal, and term `t`, in years. The result is a percentage of the
    loan principal that needs to be payed monthly for the duration of `t` years
    to fully repay the loan.
    """
    r = r / 12  # convert to monthly interest rate
    t = t * 12  # convert to months
    return r * np.power(1 + r, t) / (np.power(1 + r, t) - 1)


def get_payment(p: R, r: R, t: R) -> R:
    return p * get_rate(r, t)


def get_loan_term(p: R, a: R, r: R) -> R:
    """
    Calculates loan term of an amortized loan with loan principal `p`, monthly
    interest rate `r` and monthly payment `a`. `get_n` uses the same equation
    as `get_rate` but rewritten to express the loan term in months.
    """
    return np.log(1 - r * p / a) / np.log(1 / (1 + r))


def principals(p: R, r: R, t: R) -> np.ndarray:
    """
    Calculates rate at which the principal `p` decreases as a function of time
    for the loan with loan term `t` years and interest rate `r`.
    """
    am: R = p * get_rate(r, t)  # monthly payment (i.e., interest + amortization)
    rm: R = r / 12  # convert to monthly interest rate
    tm: N = int(t * 12)

    ps: list[R] = [p]
    for _ in range(tm):
        pn = p + p * rm - am
        if pn > p:
            raise ValueError
        else:
            pass
        ps.append(pn)
        p = pn

    return np.array(ps)


def loan_dynamics(p: R, r: R, t: R) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes an amortizing loan with principal `p`, interest rate `r` and
    loan term `t` in three components: remaining principal, interest payment
    and amortization payment.
    """
    am: R = p * get_rate(r, t)
    rm: R = r / 12

    pms: np.ndarray = principals(p, r, t)
    rms = rm * pms
    ams = am - rms

    return pms, rms, ams


def _discount(rs: tuple[np.ndarray, np.ndarray, np.ndarray], rf: R) -> R:
    """
    Computes continuously discounted present value of future loan payments with
    the flat rate `rf`.
    """
    t_start, t_end = 1 / 12, rs[0].size / 12
    t = np.linspace(t_start, t_end, rs[0].size)

    discount_factors = np.exp(-rf * t)

    pms, rms, ams = rs * discount_factors

    return pms, rms, ams


def discounted_loan_dynamics(
    p: R, r: R, t: R, rf: R
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _discount(loan_dynamics(p, r, t), rf)


def total_payment_wrt_time(
    r: R, t0: int, t1: int, rf: R = 0
) -> list[np.ndarray, np.ndarray]:
    """
    Compute the change of the total loan cost (i.e., interest + amortization)
    for a given interest rate for loan terms between t0 and t1, given in years.
    """
    p = 1
    xs, ts = [], []
    for t in range(t0, t1):
        pms, rms, ams = discounted_loan_dynamics(p, r, t, rf)
        xs.append(np.sum(rms[:-1] + ams[:-1]) / p)
        ts.append(t)

    return np.array(ts), np.array(xs)
