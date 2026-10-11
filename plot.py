# plot.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Configuration
# =========================

PREDICTION_PATH = "./results/test_predictions.csv"
SAVE_DIR = "./results/plots"

SLOPE_HEIGHT = 50.0
SLOPE_ANGLE = 60.0

FORECAST_STEPS = [1, 5, 10, 15]

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 2. Read prediction results
# =========================

df = pd.read_csv(PREDICTION_PATH)

# Use isclose to avoid floating-point problems
case_df = df[
    np.isclose(
        df["SlopeHeight_m"],
        SLOPE_HEIGHT,
        atol=1e-3
    ) &
    np.isclose(
        df["SlopeAngle_deg"],
        SLOPE_ANGLE,
        atol=1e-3
    )
].copy()

if case_df.empty:
    raise ValueError(
        f"No data found for H={SLOPE_HEIGHT} m, "
        f"Angle={SLOPE_ANGLE}°"
    )


# =========================
# 3. Metric calculation
# =========================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    error = predicted - actual

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    # MAPE
    mape = np.mean(
        np.abs(error / actual)
    ) * 100

    # R²
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return mae, rmse, mape, r2


# =========================
# 4. Create figure
# =========================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# =========================
# 5. Plot each forecast step
# =========================

for ax, step in zip(axes, FORECAST_STEPS):

    step_df = case_df[
        case_df["Forecast_Step"] == step
    ].sort_values("Sample")

    if step_df.empty:
        ax.set_title(
            f"Forecast Step = {step}"
        )
        continue

    actual = step_df[
        "Actual_Settlement_mm"
    ].values

    predicted = step_df[
        "Predicted_Settlement_mm"
    ].values

    # Calculate metrics
    mae, rmse, mape, r2 = calculate_metrics(
        actual,
        predicted
    )

    # -------------------------
    # Actual curve
    # -------------------------
    ax.plot(
        step_df["Sample"],
        actual,
        label="Actual",
        linewidth=1.5
    )

    # -------------------------
    # Predicted curve
    # -------------------------
    ax.plot(
        step_df["Sample"],
        predicted,
        label="Predicted",
        linewidth=1.5,
        linestyle="--"
    )

    # -------------------------
    # Subplot title
    # -------------------------
    ax.set_title(
        f"Forecast Step = {step}",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Sample",
        fontsize=10
    )

    ax.set_ylabel(
        "Settlement (mm)",
        fontsize=10
    )

    # -------------------------
    # Grid
    # -------------------------
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5
    )

    # -------------------------
    # Evaluation box
    # -------------------------
    metrics_text = (
        f"MAE  = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"MAPE = {mape:.2f}%\n"
        f"R²     = {r2:.4f}"
    )

    ax.text(
        0.97,
        0.05,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.85
        )
    )

    # -------------------------
    # Legend
    # -------------------------
    ax.legend(
        loc="upper left",
        fontsize=9
    )


# =========================
# 6. Overall figure
# =========================

fig.suptitle(
    f"Settlement Prediction: "
    f"H = {SLOPE_HEIGHT:.0f} m, "
    f"Angle = {SLOPE_ANGLE:.0f}°",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# =========================
# 7. Save
# =========================

save_path = os.path.join(
    SAVE_DIR,
    f"prediction_H{int(SLOPE_HEIGHT)}_"
    f"Angle{int(SLOPE_ANGLE)}.png"
)

plt.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {save_path}")