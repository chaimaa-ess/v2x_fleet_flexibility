
from pathlib import Path

import numpy as np
import pandas as pd
import picos as pc
import matplotlib.pyplot as plt
import pickle

DT = 0.25
T = 96

PI_IMB = np.array([50, 150, 200, 250, 750], dtype=float)
PI_DOE = np.array([250, 750, 1000], dtype=float)

SOLVER = "gurobi"
VERBOSITY = 0
TOL = 1e-5

PREDICTIONS_PATH = Path("results/final_forecasts/residential/k48/random_forest/predictions_test.npz")
A_PATH = Path("A_res.pkl")

OUTPUT_DIR = Path("doe_case_study")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#  tariff arrays, are in pence/kWh.
PRICE_IMPORT = [
    13.86, 13.86, 18.9, 18.9, 15.81, 15.81, 14.8, 14.8,
    15.81, 15.81, 14.78, 14.78, 16.59, 16.59, 15.75, 15.75,
    17.22, 17.22, 15.81, 15.81, 17.85, 17.85, 17.67, 17.67,
    22.63, 22.63, 22.44, 22.44, 23.1, 23.1, 25.41, 25.41,
    24.57, 24.57, 24.78, 24.78, 23.08, 23.08, 22.47, 22.47,
    20.58, 20.58, 19.53, 19.53, 17.85, 17.85, 17.84, 17.84,
    18.42, 18.42, 18.94, 18.94, 19.78, 19.78, 19.38, 19.38,
    21.82, 21.82, 21.82, 21.82, 22.26, 22.26, 25.2, 25.2,
    37.8, 37.8, 40.82, 40.82, 38.57, 38.57, 39.9, 39.9,
    40.82, 40.82, 40.32, 40.32, 25.35, 25.35, 24.15, 24.15,
    22.68, 22.68, 23.1, 23.1, 21.69, 21.69, 21.61, 21.61,
    20.66, 20.66, 17.89, 17.89, 21.84, 21.84, 18.96, 18.96,
]

PRICE_EXPORT_48 = [
    6.98, 9.24, 7.86, 7.41, 7.86, 7.4, 8.21, 7.83,
    8.49, 7.86, 8.77, 8.69, 10.91, 10.83, 11.12, 12.15,
    11.78, 11.87, 11.11, 10.84, 9.99, 9.52, 8.77, 8.76,
    9.02, 9.26, 9.63, 9.46, 10.55, 10.55, 10.74, 12.06,
    17.39, 18.74, 17.73, 18.33, 18.74, 18.52, 12.13, 11.59,
    10.93, 11.12, 10.49, 10.45, 10.03, 8.79, 10.56, 9.27,
]

PRICE_EXPORT = np.repeat(PRICE_EXPORT_48, 2)

lambda_import = np.asarray(PRICE_IMPORT, dtype=float) / 100
lambda_export = np.asarray(PRICE_EXPORT, dtype=float) / 100

with open(A_PATH, "rb") as f:
    A = pickle.load(f)

A = np.asarray(A, dtype=float)
assert A.shape == (6 * T, 2 * T)

assert lambda_import.shape == (T,)
assert lambda_export.shape == (T,)

export_limit = np.zeros(T)
import_limit = np.zeros(T)

# Export capacity (positive magnitudes, kW)
export_limit[0:20] = 250
export_limit[20:44] = 500
export_limit[44:56] = 0
export_limit[56:68] = 250
export_limit[68:96] = 500

# Import capacity (kW)
import_limit[0:20] = 250
import_limit[20:44] = 100
import_limit[44:56] = 500
import_limit[56:68] = 250
import_limit[68:96] = 100

# Signed net-power bounds
doe_export = -export_limit
doe_import = import_limit

assert doe_import.shape == (96,)
assert doe_export.shape == (96,)
assert np.all(doe_export <= doe_import)

# Expected forecast target order:
# [Pmax, Cmax, Cmin]
#
# Confirm against the target_names stored in your NPZ.
TARGET_ORDER = ("Pmax", "Cmax", "Cmin")

####################################################
# Load eligible full-day forecasts                 #
"""This selects only forecast windows starting at midnight. It does not treat the overlapping 15-minute windows as independent days."""
####################################################
def load_full_day_forecasts(path):
    with np.load(path, allow_pickle=False) as data:
        actual = data["y_true"].astype(float)
        forecast = data["y_pred"].astype(float)

        target_start = pd.DatetimeIndex(
            data["target_start"]
        )
        origin = pd.DatetimeIndex(data["origin"])
        target_names = tuple(data["target_names"].tolist())
        lead = int(data["k"])

    if lead != 48:
        raise ValueError(f"Expected k=48, found k={lead}")

    if actual.shape != forecast.shape:
        raise ValueError("Actual and forecast shapes differ.")

    if actual.shape[1:] != (T, 3):
        raise ValueError(
            f"Expected (n_samples, {T}, 3), got {actual.shape}"
        )

    print("Stored target names:", target_names)

    # Select exactly midnight-to-23:45 forecasts.
    mask = (
        (target_start == target_start.normalize())
        & (origin < target_start)
    )

    dates = target_start[mask]
    actual = actual[mask]
    forecast = forecast[mask]

    if not dates.is_unique:
        raise ValueError(
            "Multiple forecasts found for the same target day."
        )

    if not np.isfinite(actual).all():
        raise ValueError("Non-finite actual target values.")

    if not np.isfinite(forecast).all():
        raise ValueError("Non-finite forecast target values.")

    print("Eligible full days:", len(dates))

    return dates, actual, forecast, target_names

#############################################################
###       Construct the aggregate flexibility polytope    ###
#############################################################
def build_aggregate_rhs(
    target_array,
    target_names,
):
    """
    Construct the aggregate polytope RHS.

    Expected:
        Pmax, Cmax, Cmin

    b = [
        Pmax,
        zeros,
        zeros,
        -Pmax,
        Cmax,
        Cmin
    ]
    """

    x = np.asarray(target_array, dtype=float)

    if x.shape != (T, 3):
        raise ValueError(
            f"Expected shape {(T, 3)}, got {x.shape}"
        )

    names = [str(name).lower() for name in target_names]

    required = ["pmax", "cmax", "cmin"]

    if set(names) != set(required):
        raise ValueError(
            f"Unexpected target names: {target_names}. "
            "Update the target mapping before proceeding."
        )

    values = {
        name: x[:, j]
        for j, name in enumerate(names)
    }

    pmax = values["pmax"]
    cmax = values["cmax"]
    cmin = values["cmin"]

    return np.concatenate([
        pmax,
        np.zeros(T),
        np.zeros(T),
        pmax, #-pmax,
        cmax,
        cmin,
    ])

#####################################
###  Common optimization helpers  ###
#####################################
def electricity_bill(p, lam_imp, lam_exp, dt=DT):
    """
    Electricity bill in GBP.
    Prices are in GBP/kWh.
    """
    p = np.asarray(p, dtype=float).reshape(-1)

    return float(
        dt * (
            lam_imp @ p[:T]
            + lam_exp @ p[T:]
        )
    )


def net_power(p):
    p = np.asarray(p, dtype=float).reshape(-1)
    return p[:T] + p[T:]


def cost_expression(p, lam_imp_pc, lam_exp_pc):
    return DT * (
        (lam_imp_pc | p[:T])
        + (lam_exp_pc | p[T:])
    )


def solve_lp(problem, solver=SOLVER, verbosity=VERBOSITY):
    try:
        solution = problem.solve(
            solver=solver,
            verbosity=verbosity,
        )

        status = str(problem.status)

        if status != "optimal":
            return False, status

        return True, str(solution.claimedStatus)

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_rhs(b):
    b = np.asarray(b, dtype=float).reshape(-1)

    if b.shape != (6 * T,):
        raise ValueError(
            f"Expected b shape {(6*T,)}, got {b.shape}"
        )

    if not np.isfinite(b).all():
        raise ValueError("Non-finite polytope RHS.")

    return b

#######################################################
### Day-ahead scheduling under forecast flexibility ###
########################################################
def solve_forecast_day_ahead(
    A_agg,
    b_forecast,
    lambda_import,
    lambda_export,
    doe_import,
    doe_export,
    solver="gurobi",
    verbosity=0,
):
    """
    Day-ahead scheduling using forecast flexibility.

    p_ch  >= 0
    p_dis <= 0
    p_net = p_ch + p_dis

    Sign restrictions are assumed to be encoded in A.
    """

    b_forecast = validate_rhs(b_forecast)

    problem = pc.Problem()

    p = pc.RealVariable("p_plan", 2 * T)

    A_pc = pc.Constant("A", A_agg)
    b_pc = pc.Constant("b_forecast", b_forecast)

    lam_imp = pc.Constant(
        "lambda_import", lambda_import
    )
    lam_exp = pc.Constant(
        "lambda_export", lambda_export
    )

    doe_imp = pc.Constant("doe_import", doe_import)
    doe_exp = pc.Constant("doe_export", doe_export)

    p_net = p[:T] + p[T:]

    problem.add_constraint(A_pc * p <= b_pc)
    problem.add_constraint(p_net <= doe_imp)
    problem.add_constraint(p_net >= doe_exp)

    problem.set_objective(
        "min",
        cost_expression(p, lam_imp, lam_exp),
    )

    ok, status = solve_lp(
        problem, solver, verbosity
    )

    if not ok:
        return {
            "feasible": False,
            "status": status,
        }

    p_value = np.asarray(
        p.value, dtype=float
    ).reshape(-1)

    return {
        "feasible": True,
        "status": status,
        "p": p_value,
        "p_net": net_power(p_value),
        "bill_gbp": electricity_bill(
            p_value,
            lambda_import,
            lambda_export,
        ),
    }

###############################################################
### Corrective scheduling with mutually exclusive penalties ###
###############################################################
def solve_corrective_schedule(
    A_agg,
    b_actual,
    p_plan,
    lambda_import,
    lambda_export,
    doe_import,
    doe_export,
    pi_imb,
    pi_doe,
    solver="gurobi",
    verbosity=0,
):
    """
    Corrective scheduling under actual flexibility.

    pi_imb and pi_doe are in GBP/MWh.

    Non-overlapping penalty convention:
        deviation inside DOE -> imbalance rate
        deviation outside DOE -> DOE rate

    Requires:
        pi_doe >= pi_imb >= 0
    """

    if not (0 <= pi_imb <= pi_doe):
        raise ValueError(
            "Require 0 <= pi_imb <= pi_doe."
        )

    b_actual = validate_rhs(b_actual)
    p_plan = np.asarray(p_plan, dtype=float).reshape(-1)

    if p_plan.shape != (2 * T,):
        raise ValueError("Invalid planned schedule shape.")

    plan_net = net_power(p_plan)

    problem = pc.Problem()

    p = pc.RealVariable("p_real", 2 * T)

    # Absolute deviation from the planned net schedule.
    z = pc.RealVariable("deviation", T)

    # DOE exceedance magnitude.
    v = pc.RealVariable("doe_violation", T)

    A_pc = pc.Constant("A", A_agg)
    b_pc = pc.Constant("b_actual", b_actual)

    lam_imp = pc.Constant(
        "lambda_import", lambda_import
    )
    lam_exp = pc.Constant(
        "lambda_export", lambda_export
    )

    plan_pc = pc.Constant("plan_net", plan_net)
    doe_imp = pc.Constant("doe_import", doe_import)
    doe_exp = pc.Constant("doe_export", doe_export)

    p_net = p[:T] + p[T:]

    problem.add_constraint(A_pc * p <= b_pc)

    # Absolute schedule deviation.
    problem.add_constraint(z >= 0)
    problem.add_constraint(z >= p_net - plan_pc)
    problem.add_constraint(z >= plan_pc - p_net)

    # DOE violation.
    problem.add_constraint(v >= 0)
    problem.add_constraint(v >= p_net - doe_imp)
    problem.add_constraint(v >= doe_exp - p_net)

    # GBP/MWh * kW * h / 1000 = GBP.
    penalty_expression = (
        DT / 1000
    ) * (
        pi_imb * pc.sum(z)
        + (pi_doe - pi_imb) * pc.sum(v)
    )

    objective = (
        cost_expression(p, lam_imp, lam_exp)
        + penalty_expression
    )

    problem.set_objective("min", objective)

    ok, status = solve_lp(
        problem, solver, verbosity
    )

    if not ok:
        return {
            "feasible": False,
            "status": status,
        }

    p_real = np.asarray(
        p.value, dtype=float
    ).reshape(-1)

    real_net = net_power(p_real)

    # Recompute all penalty volumes from actual schedules.
    deviation_kw = np.abs(real_net - plan_net)

    violation_kw = np.maximum.reduce([
        real_net - doe_import,
        doe_export - real_net,
        np.zeros(T),
    ])

    # Check DOE violation is part of schedule deviation.
    if np.any(violation_kw > deviation_kw + 1e-4):
        raise RuntimeError(
            "DOE violation exceeds schedule deviation. "
            "Check planned DOE compliance."
        )

    imbalance_kw = np.maximum(
        deviation_kw - violation_kw,
        0.0,
    )

    imbalance_mwh = (
        DT / 1000 * np.sum(imbalance_kw)
    )

    violation_mwh = (
        DT / 1000 * np.sum(violation_kw)
    )

    deviation_mwh = (
        DT / 1000 * np.sum(deviation_kw)
    )

    bill_real = electricity_bill(
        p_real,
        lambda_import,
        lambda_export,
    )

    bill_plan = electricity_bill(
        p_plan,
        lambda_import,
        lambda_export,
    )

    delta_bill = bill_real - bill_plan

    cost_imb = pi_imb * imbalance_mwh
    cost_doe = pi_doe * violation_mwh

    delta_total = (
        delta_bill
        + cost_imb
        + cost_doe
    )

    # Verify the non-overlapping volume accounting.
    assert np.isclose(
        deviation_mwh,
        imbalance_mwh + violation_mwh,
        atol=1e-7,
    )

    return {
        "feasible": True,
        "status": status,
        "p_real": p_real,
        "p_net": real_net,
        "bill_plan_gbp": bill_plan,
        "bill_real_gbp": bill_real,
        "delta_bill_gbp": delta_bill,
        "imbalance_mwh": imbalance_mwh,
        "doe_violation_mwh": violation_mwh,
        "total_deviation_mwh": deviation_mwh,
        "cost_imb_gbp": cost_imb,
        "cost_doe_gbp": cost_doe,
        "delta_total_gbp": delta_total,
        "has_doe_violation": bool(
            violation_mwh > TOL / 1000
        ),
    }

"""
An important numerical detail: when pi_{DOE}=pi_{imb}, the auxiliary DOE variable has zero objective coefficient and may not equal the physical exceedance. 
That does not affect the reported results because the code recalculates violations directly from the realized schedule.
"""

###########################################################
### Run all eligible test days and penalty combinations ###
###########################################################
"""
We first solve each day's day-ahead problem **once**. Then we run the corrective optimization for every valid penalty combination.
"""
def run_doe_case_study(
    dates,
    actual_days,
    forecast_days,
    target_names,
    A,
    lambda_import,
    lambda_export,
    doe_import,
    doe_export,
    pi_imb_values,
    pi_doe_values,
    output_dir,
    solver="gurobi",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    schedules = {}

    for idx, date in enumerate(dates):
        day_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        b_actual = build_aggregate_rhs(
            actual_days[idx],
            target_names,
        )

        b_forecast = build_aggregate_rhs(
            forecast_days[idx],
            target_names,
        )

        plan = solve_forecast_day_ahead(
            A_agg=A,
            b_forecast=b_forecast,
            lambda_import=lambda_import,
            lambda_export=lambda_export,
            doe_import=doe_import,
            doe_export=doe_export,
            solver=solver,
        )

        if not plan["feasible"]:
            failures.append({
                "date": day_str,
                "stage": "day_ahead",
                "status": plan["status"],
            })
            continue

        schedules[day_str] = {
            "planned": plan["p"],
            "planned_net": plan["p_net"],
        }

        for pi_doe in pi_doe_values:
            for pi_imb in pi_imb_values:

                # Invalid under the convex non-overlap model.
                if pi_doe < pi_imb:
                    continue

                result = solve_corrective_schedule(
                    A_agg=A,
                    b_actual=b_actual,
                    p_plan=plan["p"],
                    lambda_import=lambda_import,
                    lambda_export=lambda_export,
                    doe_import=doe_import,
                    doe_export=doe_export,
                    pi_imb=float(pi_imb),
                    pi_doe=float(pi_doe),
                    solver=solver,
                )

                if not result["feasible"]:
                    failures.append({
                        "date": day_str,
                        "stage": "corrective",
                        "pi_imb": pi_imb,
                        "pi_doe": pi_doe,
                        "status": result["status"],
                    })
                    continue

                records.append({
                    "date": day_str,
                    "pi_imb": pi_imb,
                    "pi_doe": pi_doe,
                    **{
                        key: value
                        for key, value in result.items()
                        if not isinstance(value, np.ndarray)
                    },
                })

                schedules[day_str][
                    f"real_imb{int(pi_imb)}_doe{int(pi_doe)}"
                ] = result["p_real"]

        if (idx + 1) % 10 == 0:
            print(
                f"Completed {idx+1}/{len(dates)} days"
            )

            pd.DataFrame(records).to_csv(
                output_dir / "daily_results.csv",
                index=False,
            )

    results_df = pd.DataFrame(records)
    failures_df = pd.DataFrame(failures)

    results_df.to_csv(
        output_dir / "daily_results.csv",
        index=False,
    )

    failures_df.to_csv(
        output_dir / "failed_optimizations.csv",
        index=False,
    )

    return results_df, failures_df, schedules

## Generate the four financial heatmaps
def plot_financial_heatmaps(
    results_df,
    pi_imb_values,
    pi_doe_values,
    output_dir,
):
    metrics = [
        ("delta_total_gbp", "Total incremental cost"),
        ("delta_bill_gbp", "Electricity-bill difference"),
        ("cost_imb_gbp", "Imbalance penalty"),
        ("cost_doe_gbp", "DOE violation penalty"),
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        constrained_layout=True,
    )

    summary = (
        results_df
        .groupby(["pi_doe", "pi_imb"], as_index=False)
        .agg({
            metric: "mean"
            for metric, _ in metrics
        })
    )

    for ax, (metric, title) in zip(axes.flat, metrics):

        pivot = summary.pivot(
            index="pi_doe",
            columns="pi_imb",
            values=metric,
        ).reindex(
            index=pi_doe_values,
            columns=pi_imb_values,
        )

        values = pivot.to_numpy(dtype=float)

        masked = np.ma.masked_invalid(values)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")

        im = ax.imshow(
            masked,
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )

        ax.set_xticks(range(len(pi_imb_values)))
        ax.set_xticklabels(
            [f"{x:g}" for x in pi_imb_values]
        )

        ax.set_yticks(range(len(pi_doe_values)))
        ax.set_yticklabels(
            [f"{y:g}" for y in pi_doe_values]
        )

        ax.set_xlabel(
            r"Imbalance penalty $\pi_{\rm imb}$ (£/MWh)"
        )
        ax.set_ylabel(
            r"DOE penalty $\pi_{\rm DOE}$ (£/MWh)"
        )
        ax.set_title(title)

        # Annotate each cell with its mean daily cost.
        for i in range(len(pi_doe_values)):
            for j in range(len(pi_imb_values)):
                value = values[i, j]

                if np.isfinite(value):
                    ax.text(
                        j, i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
                else:
                    ax.text(
                        j, i,
                        "N/A",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                    )

        fig.colorbar(
            im,
            ax=ax,
            label="Mean daily cost (£/day)",
            shrink=0.85,
        )

    fig.savefig(
        Path(output_dir) / "financial_heatmaps.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        Path(output_dir) / "financial_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return summary

def check_feasibility(A, b, doe_import=None, doe_export=None):
    prob = pc.Problem()
    p = pc.RealVariable("p_check", 192)

    prob.add_constraint(
    pc.Constant("A_check", A) * p
    <= pc.Constant("b_check", np.asarray(b, dtype=float))
)
    if doe_import is not None:
        net = p[:96] + p[96:]
        prob.add_constraint(net <= pc.Constant("imp_check", doe_import))
        prob.add_constraint(net >= pc.Constant("exp_check", doe_export))

    prob.set_objective("min", pc.sum(p * 0))
    try:
        prob.solve(solver="gurobi", verbosity=2, primals=False, duals=False)
        return str(prob.status)
    except Exception as e:
        return str(e)

if __name__ == "__main__":

    dates, actual_days, forecast_days, target_names = (
        load_full_day_forecasts(PREDICTIONS_PATH)
    )

    #b = build_aggregate_rhs(forecast_days[0], target_names)


    #print("Without DOE:", check_feasibility(A, b))
    #print("With DOE:", check_feasibility(A, b, doe_import, doe_export))

    """
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("feasibility")
    p = m.addMVar(192, lb=-GRB.INFINITY)

    m.addConstr(A @ p <= b, name="flexibility")

    net = p[:96] + p[96:]   
    m.addConstr(net <= doe_import, name="DOE_import")
    m.addConstr(net >= doe_export, name="DOE_export")

    m.optimize()

    print("Gurobi status:", m.Status)

    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("infeasible.ilp")
        print("IIS saved to infeasible.ilp")
    """
    
    #dates = dates[:3]
    #actual_days = actual_days[:3]
    #forecast_days = forecast_days[:3]

    results_df, failures_df, schedules = run_doe_case_study(
        dates=dates,
        actual_days=actual_days,
        forecast_days=forecast_days,
        target_names=target_names,
        A=A,
        lambda_import=lambda_import,
        lambda_export=lambda_export,
        doe_import=doe_import,
        doe_export=doe_export,
        pi_imb_values=PI_IMB,
        pi_doe_values=PI_DOE,
        output_dir=OUTPUT_DIR,
        solver=SOLVER,
    )

    print("Successful optimizations:", len(results_df))
    print("Failed optimizations:", len(failures_df))

    if results_df.empty:
        raise RuntimeError(
            "No successful optimization results. "
            "Check failed_optimizations.csv."
        )

    heatmap_summary = plot_financial_heatmaps(
        results_df,
        PI_IMB,
        PI_DOE,
        OUTPUT_DIR,
    )

    scenario_summary = (
        results_df
        .groupby(["pi_imb", "pi_doe"])
        .agg(
            days=("date", "nunique"),
            mean_total_gbp=("delta_total_gbp", "mean"),
            mean_delta_bill_gbp=("delta_bill_gbp", "mean"),
            mean_imbalance_gbp=("cost_imb_gbp", "mean"),
            mean_doe_gbp=("cost_doe_gbp", "mean"),
            mean_imbalance_mwh=("imbalance_mwh", "mean"),
            mean_doe_violation_mwh=(
                "doe_violation_mwh", "mean"
            ),
            violation_day_rate=(
                "has_doe_violation", "mean"
            ),
        )
        .reset_index()
    )

    scenario_summary.to_csv(
        OUTPUT_DIR / "scenario_summary.csv",
        index=False,
    )

    print(scenario_summary.to_string(index=False))
