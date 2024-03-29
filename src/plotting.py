from pathlib import Path

from numbers import Integral as N
from numbers import Real as R

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.fair_price_model import amortized_matching_T, annuity_matching_T
from src.annuity_loan_model import discounted_loan_dynamics, total_payment_wrt_time


def plot_total_payment_wrt_time(rates: list[R], rf: R = 0) -> None:
    """
    Plots how total loan cost (i.e., interest + amortization), relative to the
    loan principal, changes with respect to the changing loan term. `rf` is the
    flat discount rate and for non-zero value of `rf` future loan payments are
    discounted.
    """
    _T_START, _T_END = 4, 31
    for rate in rates:
        ts, xs = total_payment_wrt_time(rate, _T_START, _T_END, rf)
        dt, dx = np.diff(ts), np.diff(xs)
        dxdt = dx[0] / dt[0]
        plt.plot(ts[1:], xs[1:], label=f"r={rate:.2%}")
        plt.plot(ts[1:], xs[0] + dxdt * np.cumsum(np.diff(ts)), "k--", linewidth=1.0)

    XLABEL, YLABEL = "t [years]", "$\eta$"
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.grid()
    plt.legend()

    plt.show()


def plot_loan_dynamics(p: R, r: R, t: R, rf: R = 0.00) -> None:
    """
    Plots figures showing loan dynamics for the loan with principal `p`,
    interest rate `r` and term `t`. `rf` is the flat discount rate for
    discounting future loan payments.
    """
    nt: N = int(t * 12)
    pms, rms, ams = discounted_loan_dynamics(p, r, t, rf)
    payment = rms + ams

    CCY = "SEK"
    START_DATE = "2023-01-01"

    ts = mdates.date2num(pd.date_range(start=START_DATE, periods=nt, freq="M"))

    _, ax = plt.subplots(1, 1)

    TIME_LABEL = "Date"

    _base = 2 if t < 20 else 5

    ax.set_title(f"P: {p:,.0f} {CCY}; Interest rate: {r:.2%}; Loan term: {t} years")
    ax.plot(ts, pms[:-1], "k-")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(f"Outstanding principal [{CCY}]")
    ax.grid()

    _, ax = plt.subplots(1, 1)

    COLOR1, ALPHA1 = "blue", 0.20
    COLOR2, ALPHA2 = "lightblue", 0.50
    LEGEND_HANDLES = [
        patches.Patch(facecolor=COLOR1, alpha=ALPHA1, label="Interest"),
        patches.Patch(facecolor=COLOR2, alpha=ALPHA2, label="Amortization"),
    ]

    PAYMENT_LABEL = f"Payments [{CCY}]"

    ax.set_title(
        f"Payment = {round(rms[0] + ams[0], -2):,.0f} SEK; Interest / Amortization = {rms[:-1].sum()/ams[:-1].sum():.2%}"
    )
    ax.plot(ts, rms[:-1], "k-.")
    ax.plot(ts, payment[:-1], "k")
    ax.fill_between(
        x=ts, y1=rms[:-1], y2=np.full(rms[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax.fill_between(x=ts, y1=payment[:-1], y2=rms[:-1], color=COLOR2, alpha=ALPHA2)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, (ams[0] + rms[0]) * (1 + 0.3))
    ax.grid()
    ax.legend(handles=LEGEND_HANDLES, loc=1)

    _, ax = plt.subplots(1, 1)

    ax.plot(ts, rms[:-1] / pms[:-1], "k-.", label="r")
    ax.plot(ts, ams[:-1] / pms[:-1], "k", label="$A/P(t) - r$")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel(TIME_LABEL)
    ax.set_yscale("log")
    locator = mticker.FixedLocator([0.01, 0.1, 1])
    ax.yaxis.set_major_locator(locator)
    formatter = mticker.FixedFormatter(["1%", "10%", "100%"])
    ax.yaxis.set_major_formatter(formatter)
    ax.grid()
    ax.legend(loc=2)

    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r:.2%}; Term: {t} Y => {(payment[:-1].sum()-p)/p:.2%}"
    )

    plt.tight_layout()
    plt.show()


# plot_discount_loan_dynamics(2_000_000, 0.03, 20)


def plot_rate_difference(p: R, r1: R, r2: R, r3: R, t: R, rf: R = 0.00) -> None:
    """
    Plots figures showing loan sensitivity for loan interest rates `r1`, `r2`
    and `r3`, when loan principal is `p` and loan term is `t`.
    """
    nt: N = int(t * 12)
    pms1, rms1, ams1 = discounted_loan_dynamics(p, r1, t, rf)
    payment1 = rms1 + ams1
    pms2, rms2, ams2 = discounted_loan_dynamics(p, r2, t, rf)
    payment2 = rms2 + ams2
    pms3, rms3, ams3 = discounted_loan_dynamics(p, r3, t, rf)
    payment3 = rms3 + ams3

    CCY = "SEK"
    START_DATE = "2023-01-01"

    ts = mdates.date2num(pd.date_range(start=START_DATE, periods=nt, freq="M"))

    _, ax1 = plt.subplots(1, 1)

    TIME_LABEL = "Date"

    _base = 2 if t < 20 else 5

    ax1.set_title(f"P: {p:,.0f} {CCY}; Loan term: {t} years")
    ax1.plot(ts, pms1[:-1], "k-.", label=f"{r1:.2%}")
    ax1.plot(ts, pms2[:-1], "k--", label=f"{r2:.2%}")
    ax1.plot(ts, pms3[:-1], "k", label=f"{r3:.2%}")
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, p * (1 + 0.05))
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel(f"Outstanding principal [{CCY}]")
    ax1.grid()
    ax1.legend(loc=1)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    COLOR1, ALPHA1 = "blue", 0.20
    COLOR2, ALPHA2 = "lightblue", 0.50
    LEGEND_HANDLES = [
        patches.Patch(facecolor=COLOR1, alpha=ALPHA1, label="Interest"),
        patches.Patch(facecolor=COLOR2, alpha=ALPHA2, label="Amortization"),
    ]

    PAYMENT_LABEL = f"Payments [{CCY}]"

    Y_LIMIT = (rms3[0] + ams3[0]).sum() * (1 + 0.1)

    ax1.set_title(
        f"Rate: {r1:.2%} \n Payment = {round(rms1[0] + ams1[0], -2):,.0f} SEK; Interest / Amortization = {rms1[:-1].sum()/ams1[:-1].sum():.2%}"
    )
    ax1.plot(ts, rms1[:-1], "k-.")
    ax1.plot(ts, payment1[:-1], "k")
    ax1.fill_between(
        x=ts, y1=rms1[:-1], y2=np.full(rms1[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax1.fill_between(x=ts, y1=payment1[:-1], y2=rms1[:-1], color=COLOR2, alpha=ALPHA2)
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, Y_LIMIT)
    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel(PAYMENT_LABEL)
    ax1.grid()
    ax1.legend(handles=LEGEND_HANDLES, loc=1)

    ax2.set_title(
        f"Rate: {r2:.2%} \n Payment: {round(rms2[0] + ams2[0], -2):,.0f} SEK; Interest / Amortization = {rms2[:-1].sum()/ams2[:-1].sum():.2%}"
    )
    ax2.plot(ts, rms2[:-1], "k-.")
    ax2.plot(ts, payment2[:-1], "k")
    ax2.fill_between(
        x=ts, y1=rms2[:-1], y2=np.full(rms2[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax2.fill_between(x=ts, y1=payment2[:-1], y2=rms2[:-1], color=COLOR2, alpha=ALPHA2)
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_ylim(0, Y_LIMIT)
    ax2.set_xlabel(TIME_LABEL)
    ax2.grid()
    ax2.legend(handles=LEGEND_HANDLES, loc=1)

    ax3.set_title(
        f"Rate: {r3:.2%} \n Payment: {round(rms3[0] + ams3[0], -2):,.0f} SEK; Interest / Amortization = {rms3[:-1].sum()/ams3[:-1].sum():.2%}"
    )
    ax3.plot(ts, rms3[:-1], "k-.")
    ax3.plot(ts, payment3[:-1], "k")
    ax3.fill_between(
        x=ts, y1=rms3[:-1], y2=np.full(rms3[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax3.fill_between(x=ts, y1=payment3[:-1], y2=rms3[:-1], color=COLOR2, alpha=ALPHA2)
    ax3.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.tick_params(axis="x", rotation=45)
    ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax3.set_ylim(0, Y_LIMIT)
    ax3.set_xlabel(TIME_LABEL)
    ax3.grid()
    ax3.legend(handles=LEGEND_HANDLES, loc=1)

    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r1:.2%}; Term: {t} Y => {(payment1[:-1].sum()-p)/p:.2%}"
    )
    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r2:.2%}; Term: {t} Y => {(payment2[:-1].sum()-p)/p:.2%}"
    )
    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r3:.2%}; Term: {t} Y => {(payment3[:-1].sum()-p)/p:.2%}"
    )

    plt.tight_layout()
    plt.show()


# plot_rate_difference(2_000_000, 3_000_000, 0.015, 0.03, 0.045, 30)


def plot_term_difference(p: R, r: R, t1: R, t2: R, t3: R, rf: R = 0.00) -> None:
    """
    Plots figures showing loan sensitivity for loan terms `t1`, `t2` and `t3`,
    when the loan principal is `p` and loan interest rate is `r`.
    """
    nt1: N = int(t1 * 12)
    nt2: N = int(t2 * 12)
    nt3: N = int(t3 * 12)
    pms1, rms1, ams1 = discounted_loan_dynamics(p, r, t1, rf)
    payment1 = rms1 + ams1
    pms2, rms2, ams2 = discounted_loan_dynamics(p, r, t2, rf)
    payment2 = rms2 + ams2
    pms3, rms3, ams3 = discounted_loan_dynamics(p, r, t3, rf)
    payment3 = rms3 + ams3

    CCY = "SEK"
    START_DATE = "2023-01-01"

    ts1 = mdates.date2num(pd.date_range(start=START_DATE, periods=nt1, freq="M"))
    ts2 = mdates.date2num(pd.date_range(start=START_DATE, periods=nt2, freq="M"))
    ts3 = mdates.date2num(pd.date_range(start=START_DATE, periods=nt3, freq="M"))

    _, ax1 = plt.subplots(1, 1)

    TIME_LABEL = "Date"

    _base = 2 if t3 < 20 else 5

    ax1.set_title(f"P: {p:,.0f} {CCY}; Interest rate: {r:.2%}")
    ax1.plot(ts1, pms1[:-1], "k-.", label=f"{t1} years")
    ax1.plot(ts2, pms2[:-1], "k--", label=f"{t2} years")
    ax1.plot(ts3, pms3[:-1], "k", label=f"{t3} years")
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, p * (1 + 0.05))
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel(f"Outstanding principal [{CCY}]")
    ax1.grid()
    ax1.legend(loc=1)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    COLOR1, ALPHA1 = "blue", 0.20
    COLOR2, ALPHA2 = "lightblue", 0.50
    LEGEND_HANDLES = [
        patches.Patch(facecolor=COLOR1, alpha=ALPHA1, label="Interest"),
        patches.Patch(facecolor=COLOR2, alpha=ALPHA2, label="Amortization"),
    ]

    PAYMENT_LABEL = f"Payments [{CCY}]"

    Y_LIMIT = (rms1[0] + ams1[0]).sum() * (1 + 0.1)

    ax1.set_title(
        f"Rate: {r:.2%} \n Payment: {round(rms1[0] + ams1[0], -2):,.0f}; Interest / Amortization = {rms1[:-1].sum()/ams1[:-1].sum():.2%}"
    )
    ax1.plot(ts1, rms1[:-1], "k-.")
    ax1.plot(ts1, payment1[:-1], "k")
    ax1.fill_between(
        x=ts1, y1=rms1[:-1], y2=np.full(rms1[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax1.fill_between(x=ts1, y1=payment1[:-1], y2=rms1[:-1], color=COLOR2, alpha=ALPHA2)
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, Y_LIMIT)
    ax1.set_ylabel(PAYMENT_LABEL)
    ax1.grid()
    ax1.legend(handles=LEGEND_HANDLES, loc=1)

    ax2.set_title(
        f"Payment: {r:.2%} \n {round(rms2[0] + ams2[0], -2):,.0f}; Interest / Amortization = {rms2[:-1].sum()/ams2[:-1].sum():.2%}"
    )
    ax2.plot(ts2, rms2[:-1], "k-.")
    ax2.plot(ts2, payment2[:-1], "k")
    ax2.fill_between(
        x=ts2, y1=rms2[:-1], y2=np.full(rms2[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax2.fill_between(x=ts2, y1=payment2[:-1], y2=rms2[:-1], color=COLOR2, alpha=ALPHA2)
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=_base))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_ylim(0, Y_LIMIT)
    ax2.set_ylabel(PAYMENT_LABEL)
    ax2.grid()
    ax2.legend(handles=LEGEND_HANDLES, loc=1)

    ax3.set_title(
        f"Payment: {r:.2%} \n {round(rms3[0] + ams3[0], -2):,.0f}; Interest / Amortization = {rms3[:-1].sum()/ams3[:-1].sum():.2%}"
    )
    ax3.plot(ts3, rms3[:-1], "k-.")
    ax3.plot(ts3, payment3[:-1], "k")
    ax3.fill_between(
        x=ts3, y1=rms3[:-1], y2=np.full(rms3[:-1].size, 0), color=COLOR1, alpha=ALPHA1
    )
    ax3.fill_between(x=ts3, y1=payment3[:-1], y2=rms3[:-1], color=COLOR2, alpha=ALPHA2)
    ax3.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.tick_params(axis="x", rotation=45)
    ax3.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax3.set_ylim(0, Y_LIMIT)
    ax3.set_xlabel(TIME_LABEL)
    ax3.set_ylabel(PAYMENT_LABEL)
    ax3.grid()
    ax3.legend(handles=LEGEND_HANDLES, loc=1)

    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r:.2%}; Term: {t1} Y => {(payment1[:-1].sum()-p)/p:.2%}"
    )
    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r:.2%}; Term: {t2} Y => {(payment2[:-1].sum()-p)/p:.2%}"
    )
    print(
        f"Principal: {p:,.0f} {CCY}; Interest rate: {r:.2%}; Term: {t3} Y => {(payment3[:-1].sum()-p)/p:.2%}"
    )

    plt.show()


# plot_term_difference(2_000_000, 0.03, 10, 20, 30)


def plot_term_sensitivity():
    """Plots the theoretical loan-term sensitivities."""
    r = np.arange(0.035, 0.05, 0.0005) / 12  # interest rate [% / mo]
    ap = (
        np.arange(0.06, 0.09, 0.0005) / 12
    )  # loan principle divided by monthly payment [1]

    r, ap = np.meshgrid(r, ap)
    dtdr = np.log(1 / (r + 1)) / ((r - ap) * np.log(1 / (r + 1)) ** 2) + np.log(
        1 - r / ap
    ) / ((r + 1) * np.log(1 / (r + 1)) ** 2)
    dtdap = -r / (ap * (r - ap) * np.log(1 / (1 + r)))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    plot1 = ax1.plot_surface(
        X=r * 12,  # interest rate [% / y]
        Y=ap * 12,  # yearly payment as percentage of the loan principle `1/(P/A) = A/P`
        Z=dtdr
        / 12
        / 100
        / 100,  # rate of change of loan term as a function of interest rate [y / (0.1%/y)]
        cmap=cm.coolwarm,
    )

    fig1.colorbar(plot1, shrink=0.5, aspect=5)
    ax1.set_xlabel(r"$r$")
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2%}"))
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
    ax1.set_ylabel(r"$\rho$")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2%}"))
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.005))
    ax1.set_zlabel(r"$\partial T / \partial r$")
    ax1.view_init(elev=25, azim=-140)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")

    surf2 = ax2.plot_surface(
        X=r * 12,  # annual interest rate [%/Year]
        Y=ap * 12,
        Z=dtdap / 12 / 100 / 100,  # annual change of interest rate [y / ($/$)] = [y]
        cmap=cm.coolwarm,
    )

    fig2.colorbar(surf2, shrink=0.5, aspect=5)
    ax2.set_xlabel(r"$r$")
    ax2.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2%}"))
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.005))
    ax2.set_ylabel(r"$\rho$")
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2%}"))
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.005))
    ax2.set_zlabel(r"$\partial T / \partial \rho$")
    ax2.view_init(elev=25, azim=-140)

    plt.show()


# plot_term_sensitivity()


def plot_annuity_matching_T(
    p1: R,
    rho: np.ndarray,
    r1: R,
    r2: R,
    tau1: R,
    tau2: R,
    deltas: np.ndarray,
    gamma: R,
    d: R,
) -> None:
    s1s = p1 / rho

    s1s, deltas = np.meshgrid(s1s, deltas)
    p2 = annuity_matching_T(p1, s1s, r1, r2, tau1, tau2, deltas, gamma)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    plot1 = ax1.plot_surface(X=p1 / s1s, Y=deltas, Z= (p2 - p1) / p1 * (1 - d), cmap=cm.coolwarm_r)

    cbar_locator = mticker.MultipleLocator(base=0.04)
    cbar_formatter = mticker.StrMethodFormatter("{x:,.0%}")
    colorbar = fig1.colorbar(plot1, shrink=0.6, aspect=5, pad=0.125, format=cbar_formatter, ticks=cbar_locator)
    ax1.set_xlabel(r"$\rho=P/S$")
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax1.set_ylabel(r"$\delta$")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax1.set_zlabel(r"$\mathrm{d}h/h$")
    ax1.zaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0%}"))
    ax1.zaxis.set_major_locator(mticker.MultipleLocator(base=0.050))

    ax1.view_init(elev=20, azim=-35)

    plt.tight_layout()
    plt.show()


"""
import numpy as np

from plotting import plot_annuity_matching_T

gamma, tau1, tau2, deltas, d = 0.02, 0.30, 0.30, np.arange(0.25, 1.01, 0.01), 0.15
p1, r1, rhos = 2_000_000, 0.015, np.arange(2, 5.5, 0.05)
r2, s2 = 0.050, 500_000 * (1 + gamma)

plot_annuity_matching_T(p1, rhos, r1, r2, tau1, tau2, deltas, gamma, d)
"""


def plot_amortized_matching_T(
    p1: R,
    rho: np.ndarray,
    r1: R,
    r2: R,
    tau1: R,
    tau2: R,
    deltas: np.ndarray,
    gamma: R,
    d: R,
) -> None:
    s1s = p1 / rho

    s1s, deltas = np.meshgrid(s1s, deltas)
    p2 = amortized_matching_T(p1, s1s, r1, r2, tau1, tau2, deltas, gamma)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    plot1 = ax1.plot_surface(X=p1 / s1s, Y=deltas, Z= (p2 - p1) / p1 * (1 - d), cmap=cm.coolwarm_r)

    cbar_locator = mticker.MultipleLocator(base=0.02)
    cbar_formatter = mticker.StrMethodFormatter("{x:,.0%}")
    colorbar = fig1.colorbar(plot1, shrink=0.6, aspect=5, pad=0.125, format=cbar_formatter, ticks=cbar_locator)
    ax1.set_xlabel(r"$\rho=P/S$")
    ax1.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax1.set_ylabel(r"$\delta$")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax1.set_zlabel(r"$\mathrm{d}h/h$")
    ax1.zaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0%}"))
    ax1.zaxis.set_major_locator(mticker.MultipleLocator(base=0.050))

    ax1.view_init(elev=20, azim=-35)

    plt.tight_layout()
    plt.show()


"""
import numpy as np

from plotting import plot_amortized_matching_T

gamma, tau1, tau2, deltas, d = 0.02, 0.30, 0.30, np.arange(0.25, 1.01, 0.01), 0.15
p1, r1, rhos = 2_000_000, 0.015, np.arange(2, 5.5, 0.05)
r2, s2 = 0.050, 500_000 * (1 + gamma)

plot_amortized_matching_T(p1, rhos, r1, r2, tau1, tau2, deltas, gamma, d)
"""


def plot_interest_rate_vs_home_prices() -> None:
    script_path = Path(__file__).resolve()
    data_path = script_path.parent.parent
    data_location = data_path / "data/bbg.csv"

    df = pd.read_csv(data_location, parse_dates=["Date"])
    ts, rates, prices = np.split(df.values, df.shape[1], axis=1)

    fig, ax1 = plt.subplots()

    ax1.plot(ts, rates, color=(color1 := "red"))
    ax1.set_xlabel("Date")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("Riskbank's policy rate [%]", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()

    ax2.plot(ts, prices, color=(color2 := "blue"))
    ax2.set_ylabel("Sweden Real Estate Pricing Index (inverted)", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout()

    plt.show()
