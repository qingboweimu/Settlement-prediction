import os
import random
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import LSTMSettlementModel


# ============================================================
# 1. Basic Settings
# ============================================================

DATA_PATH = "./tower_settlement_data.csv"

SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "best_lstm_settlement.pth"
)

SCALER_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "scalers.pkl"
)

PREDICTION_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "test_predictions.csv"
)

METRICS_CASE_PATH = os.path.join(
    SAVE_DIR,
    "test_metrics_by_case.csv"
)

METRICS_SUMMARY_PATH = os.path.join(
    SAVE_DIR,
    "test_metrics_summary.csv"
)

HISTORY_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "training_history.csv"
)


# ============================================================
# 2. Model Parameters
# ============================================================

PRED_LEN = 15

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. Dataset
# ============================================================

class SettlementDataset(Dataset):
    """
    Current time t:

        SlopeHeight
        SlopeAngle
        Settlement_t

    Future t+1:t+15:

        Rain
        Evap

    Target:

        Settlement[t+1:t+15]

    static_input:
        [3]

    future_input:
        [15, 2]

    target:
        [15, 1]
    """

    def __init__(self, data, pred_len=15):

        self.pred_len = pred_len
        self.samples = []

        grouped = data.groupby(
            [
                "SlopeHeight_m",
                "SlopeAngle_deg"
            ],
            sort=False
        )

        for _, group in grouped:

            group = group.reset_index(drop=True)

            for i in range(
                len(group) - pred_len
            ):

                self.samples.append(
                    (group, i)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        group, i = self.samples[idx]

        # ----------------------------------------
        # Current state
        # ----------------------------------------

        static_input = group.loc[
            i,
            [
                "SlopeHeight_m",
                "SlopeAngle_deg",
                "Settlement_mm"
            ]
        ].values.astype(np.float32)

        # ----------------------------------------
        # Future meteorological conditions
        # ----------------------------------------

        future_data = group.iloc[
            i + 1:
            i + 1 + self.pred_len
        ]

        future_input = future_data[
            [
                "Rain_mm",
                "Evap_mm"
            ]
        ].values.astype(np.float32)

        # ----------------------------------------
        # Future settlement
        # ----------------------------------------

        target = future_data[
            ["Settlement_mm"]
        ].values.astype(np.float32)

        # ----------------------------------------
        # Convert to Tensor
        # ----------------------------------------

        static_input = torch.tensor(
            static_input,
            dtype=torch.float32
        )

        future_input = torch.tensor(
            future_input,
            dtype=torch.float32
        )

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return (
            static_input,
            future_input,
            target
        )


# ============================================================
# 5. Load Data
# ============================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

data = pd.read_csv(DATA_PATH)

print("Data shape:", data.shape)

print(
    "Columns:",
    data.columns.tolist()
)


# ============================================================
# 6. Check Data
# ============================================================

required_columns = [
    "SlopeHeight_m",
    "SlopeAngle_deg",
    "Rain_mm",
    "Evap_mm",
    "Settlement_mm"
]

missing_columns = [
    col
    for col in required_columns
    if col not in data.columns
]

if len(missing_columns) > 0:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


missing_values = data[
    required_columns
].isnull().sum()

print("\nMissing values:")
print(missing_values)

if missing_values.sum() > 0:

    raise ValueError(
        "Dataset contains missing values."
    )


# ============================================================
# 7. Identify 16 Slope Cases
# ============================================================

cases = data[
    [
        "SlopeHeight_m",
        "SlopeAngle_deg"
    ]
].drop_duplicates()

print("\nNumber of slope cases:", len(cases))

print(cases)

if len(cases) != 16:

    print("\nWarning:")
    print(
        "Expected 16 slope cases, but found",
        len(cases)
    )


# ============================================================
# 8. Split Data
#    Each slope case independently:
#    70% Train
#    15% Validation
#    15% Test
# ============================================================

train_parts = []
val_parts = []
test_parts = []

grouped = data.groupby(
    [
        "SlopeHeight_m",
        "SlopeAngle_deg"
    ],
    sort=False
)

print("\n")
print("=" * 60)
print("Splitting data...")
print("=" * 60)


for (height, angle), group in grouped:

    group = group.reset_index(drop=True)

    n = len(group)

    if n <= PRED_LEN:

        raise ValueError(
            f"Case Height={height}, "
            f"Angle={angle} has only {n} rows."
        )

    # ----------------------------------------
    # 70% / 15% / 15%
    # ----------------------------------------

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n *
        (
            TRAIN_RATIO
            + VAL_RATIO
        )
    )

    train_group = group.iloc[
        :train_end
    ].copy()

    val_group = group.iloc[
        train_end:val_end
    ].copy()

    test_group = group.iloc[
        val_end:
    ].copy()

    # ----------------------------------------
    # Check length
    # ----------------------------------------

    if len(train_group) <= PRED_LEN:

        raise ValueError(
            f"Train data too short: "
            f"Height={height}, "
            f"Angle={angle}"
        )

    if len(val_group) <= PRED_LEN:

        raise ValueError(
            f"Validation data too short: "
            f"Height={height}, "
            f"Angle={angle}"
        )

    if len(test_group) <= PRED_LEN:

        raise ValueError(
            f"Test data too short: "
            f"Height={height}, "
            f"Angle={angle}"
        )

    train_parts.append(
        train_group
    )

    val_parts.append(
        val_group
    )

    test_parts.append(
        test_group
    )

    print(
        f"Height={height}, "
        f"Angle={angle} | "
        f"Train={len(train_group)} | "
        f"Val={len(val_group)} | "
        f"Test={len(test_group)}"
    )


train_data = pd.concat(
    train_parts,
    ignore_index=True
)

val_data = pd.concat(
    val_parts,
    ignore_index=True
)

test_data = pd.concat(
    test_parts,
    ignore_index=True
)


print("\n")
print("=" * 60)
print("Dataset summary")
print("=" * 60)

print(
    "Train rows:",
    len(train_data)
)

print(
    "Validation rows:",
    len(val_data)
)

print(
    "Test rows:",
    len(test_data)
)


# ============================================================
# 9. Fit Scalers ONLY on Training Data
# ============================================================

# Height + Angle
static_scaler = MinMaxScaler()

static_scaler.fit(
    train_data[
        [
            "SlopeHeight_m",
            "SlopeAngle_deg"
        ]
    ]
)


# Settlement
settlement_scaler = MinMaxScaler()

settlement_scaler.fit(
    train_data[
        ["Settlement_mm"]
    ]
)


# Rain + Evap
future_scaler = MinMaxScaler()

future_scaler.fit(
    train_data[
        [
            "Rain_mm",
            "Evap_mm"
        ]
    ]
)


# ============================================================
# 10. Scaling Function
# ============================================================

def scale_dataframe(
    data,
    static_scaler,
    settlement_scaler,
    future_scaler
):

    data = data.copy()

    # Height + Angle

    data[
        [
            "SlopeHeight_m",
            "SlopeAngle_deg"
        ]
    ] = static_scaler.transform(
        data[
            [
                "SlopeHeight_m",
                "SlopeAngle_deg"
            ]
        ]
    )

    # Settlement

    data[
        ["Settlement_mm"]
    ] = settlement_scaler.transform(
        data[
            ["Settlement_mm"]
        ]
    )

    # Rain + Evap

    data[
        [
            "Rain_mm",
            "Evap_mm"
        ]
    ] = future_scaler.transform(
        data[
            [
                "Rain_mm",
                "Evap_mm"
            ]
        ]
    )

    return data


train_scaled = scale_dataframe(
    train_data,
    static_scaler,
    settlement_scaler,
    future_scaler
)

val_scaled = scale_dataframe(
    val_data,
    static_scaler,
    settlement_scaler,
    future_scaler
)

test_scaled = scale_dataframe(
    test_data,
    static_scaler,
    settlement_scaler,
    future_scaler
)


# ============================================================
# 11. Create Dataset
# ============================================================

train_dataset = SettlementDataset(
    train_scaled,
    pred_len=PRED_LEN
)

val_dataset = SettlementDataset(
    val_scaled,
    pred_len=PRED_LEN
)

test_dataset = SettlementDataset(
    test_scaled,
    pred_len=PRED_LEN
)


print("\n")
print("=" * 60)
print("Sample summary")
print("=" * 60)

print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# 12. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. Check Input Shapes
# ============================================================

sample_static, sample_future, sample_target = next(
    iter(train_loader)
)

print("\n")
print("=" * 60)
print("Input shapes")
print("=" * 60)

print(
    "Static input:",
    sample_static.shape
)

print(
    "Future input:",
    sample_future.shape
)

print(
    "Target:",
    sample_target.shape
)


# ============================================================
# 14. Build Model
# ============================================================

model = LSTMSettlementModel(
    static_size=3,
    future_size=2,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pred_len=PRED_LEN
)

model = model.to(DEVICE)


print("\n")
print("=" * 60)
print("Model")
print("=" * 60)

print(model)

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# 15. Loss and Optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 16. Train Function
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for (
        static_input,
        future_input,
        target
    ) in loader:

        static_input = static_input.to(device)

        future_input = future_input.to(device)

        target = target.to(device)

        prediction = model(
            static_input,
            future_input
        )

        loss = criterion(
            prediction,
            target
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * static_input.size(0)
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    return average_loss


# ============================================================
# 17. Validation Function
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for (
            static_input,
            future_input,
            target
        ) in loader:

            static_input = static_input.to(device)

            future_input = future_input.to(device)

            target = target.to(device)

            prediction = model(
                static_input,
                future_input
            )

            loss = criterion(
                prediction,
                target
            )

            total_loss += (
                loss.item()
                * static_input.size(0)
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    return average_loss


# ============================================================
# 18. Training
# ============================================================

print("\n")
print("=" * 60)
print("Start Training")
print("=" * 60)


best_val_loss = float("inf")

patience_counter = 0

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": []
}


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )

    # ----------------------------------------
    # Save best model
    # ----------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print(
            "    Best model saved."
        )

    else:

        patience_counter += 1

    # ----------------------------------------
    # Early stopping
    # ----------------------------------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping.")

        print(
            f"Validation loss has not improved "
            f"for {PATIENCE} epochs."
        )

        break


# ============================================================
# 19. Save Training History
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_SAVE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. Save Scalers
# ============================================================

scalers = {
    "static_scaler": static_scaler,
    "settlement_scaler": settlement_scaler,
    "future_scaler": future_scaler
}

joblib.dump(
    scalers,
    SCALER_SAVE_PATH
)


# ============================================================
# 21. Load Best Model
# ============================================================

print("\n")
print("=" * 60)
print("Loading best model...")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# 22. Test Prediction
# ============================================================

all_predictions = []
all_targets = []

with torch.no_grad():

    for (
        static_input,
        future_input,
        target
    ) in test_loader:

        static_input = static_input.to(
            DEVICE
        )

        future_input = future_input.to(
            DEVICE
        )

        prediction = model(
            static_input,
            future_input
        )

        all_predictions.append(
            prediction.cpu().numpy()
        )

        all_targets.append(
            target.numpy()
        )


all_predictions = np.concatenate(
    all_predictions,
    axis=0
)

all_targets = np.concatenate(
    all_targets,
    axis=0
)


print("\n")
print("=" * 60)
print("Prediction shape")
print("=" * 60)

print(
    "Prediction:",
    all_predictions.shape
)

print(
    "Target:",
    all_targets.shape
)


# ============================================================
# 23. Inverse Transform
# ============================================================

prediction_shape = (
    all_predictions.shape
)

target_shape = (
    all_targets.shape
)


pred_original = settlement_scaler.inverse_transform(
    all_predictions.reshape(-1, 1)
).reshape(
    prediction_shape
)


target_original = settlement_scaler.inverse_transform(
    all_targets.reshape(-1, 1)
).reshape(
    target_shape
)


# ============================================================
# 24. Generate Test Prediction Records
# ============================================================

prediction_records = []

sample_index = 0


for (
    static_input,
    future_input,
    target
) in test_loader:

    batch_size = static_input.shape[0]

    for b in range(batch_size):

        # ----------------------------------------
        # Recover original Height and Angle
        # ----------------------------------------

        height_scaled = (
            static_input[b, 0].item()
        )

        angle_scaled = (
            static_input[b, 1].item()
        )

        height_original = (
            static_scaler
            .inverse_transform(
                np.array(
                    [
                        [
                            height_scaled,
                            angle_scaled
                        ]
                    ]
                )
            )[0, 0]
        )

        angle_original = (
            static_scaler
            .inverse_transform(
                np.array(
                    [
                        [
                            height_scaled,
                            angle_scaled
                        ]
                    ]
                )
            )[0, 1]
        )

        # ----------------------------------------
        # Save 15 prediction steps
        # ----------------------------------------

        for step in range(PRED_LEN):

            prediction_records.append(
                {
                    "SlopeHeight_m":
                        height_original,

                    "SlopeAngle_deg":
                        angle_original,

                    "Sample":
                        sample_index + b,

                    "Forecast_Step":
                        step + 1,

                    "Predicted_Settlement_mm":
                        pred_original[
                            sample_index + b,
                            step,
                            0
                        ],

                    "Actual_Settlement_mm":
                        target_original[
                            sample_index + b,
                            step,
                            0
                        ]
                }
            )

    sample_index += batch_size


prediction_df = pd.DataFrame(
    prediction_records
)


prediction_df.to_csv(
    PREDICTION_SAVE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. Overall Metrics
# ============================================================

y_pred = pred_original.reshape(-1)

y_true = target_original.reshape(-1)


overall_mse = mean_squared_error(
    y_true,
    y_pred
)

overall_rmse = np.sqrt(
    overall_mse
)

overall_mae = mean_absolute_error(
    y_true,
    y_pred
)

overall_r2 = r2_score(
    y_true,
    y_pred
)


# ============================================================
# 26. Case-by-Case Metrics
# ============================================================

case_metrics = []


grouped_predictions = prediction_df.groupby(
    [
        "SlopeHeight_m",
        "SlopeAngle_deg"
    ]
)


for (
    height,
    angle
), group in grouped_predictions:

    case_true = group[
        "Actual_Settlement_mm"
    ].values

    case_pred = group[
        "Predicted_Settlement_mm"
    ].values

    case_mse = mean_squared_error(
        case_true,
        case_pred
    )

    case_rmse = np.sqrt(
        case_mse
    )

    case_mae = mean_absolute_error(
        case_true,
        case_pred
    )

    case_r2 = r2_score(
        case_true,
        case_pred
    )

    case_metrics.append(
        {
            "SlopeHeight_m":
                height,

            "SlopeAngle_deg":
                angle,

            "MSE":
                case_mse,

            "RMSE":
                case_rmse,

            "MAE":
                case_mae,

            "R2":
                case_r2,

            "Num_Predictions":
                len(group)
        }
    )


case_metrics_df = pd.DataFrame(
    case_metrics
)


# ============================================================
# 27. Mean Metrics Across 16 Cases
# ============================================================

mean_case_mse = (
    case_metrics_df["MSE"].mean()
)

mean_case_rmse = (
    case_metrics_df["RMSE"].mean()
)

mean_case_mae = (
    case_metrics_df["MAE"].mean()
)

mean_case_r2 = (
    case_metrics_df["R2"].mean()
)


# ============================================================
# 28. Summary Metrics
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "Metric_Type":
                "Mean_of_16_Cases",

            "MSE":
                mean_case_mse,

            "RMSE":
                mean_case_rmse,

            "MAE":
                mean_case_mae,

            "R2":
                mean_case_r2
        },

        {
            "Metric_Type":
                "Overall_All_Test_Samples",

            "MSE":
                overall_mse,

            "RMSE":
                overall_rmse,

            "MAE":
                overall_mae,

            "R2":
                overall_r2
        }
    ]
)


case_metrics_df.to_csv(
    METRICS_CASE_PATH,
    index=False,
    encoding="utf-8-sig"
)


summary_df.to_csv(
    METRICS_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 29. Print Results
# ============================================================

print("\n")
print("=" * 60)
print("Test Results - Mean of 16 Cases")
print("=" * 60)

print(
    f"MSE  = {mean_case_mse:.6f}"
)

print(
    f"RMSE = {mean_case_rmse:.6f} mm"
)

print(
    f"MAE  = {mean_case_mae:.6f} mm"
)

print(
    f"R²   = {mean_case_r2:.6f}"
)


print("\n")
print("=" * 60)
print("Test Results - Overall")
print("=" * 60)

print(
    f"MSE  = {overall_mse:.6f}"
)

print(
    f"RMSE = {overall_rmse:.6f} mm"
)

print(
    f"MAE  = {overall_mae:.6f} mm"
)

print(
    f"R²   = {overall_r2:.6f}"
)


print("\n")
print("=" * 60)
print("Metrics by Slope Case")
print("=" * 60)

print(
    case_metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 30. Finished
# ============================================================

print("\n")
print("=" * 60)
print("Training completed.")
print("=" * 60)

print(
    "Best model:"
)

print(
    MODEL_SAVE_PATH
)

print(
    "\nScalers:"
)

print(
    SCALER_SAVE_PATH
)

print(
    "\nTest predictions:"
)

print(
    PREDICTION_SAVE_PATH
)

print(
    "\nMetrics by case:"
)

print(
    METRICS_CASE_PATH
)

print(
    "\nMetrics summary:"
)

print(
    METRICS_SUMMARY_PATH
)

print(
    "\nTraining history:"
)

print(
    HISTORY_SAVE_PATH
)

print("=" * 60)