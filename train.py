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

# ============================================================
# 从 model.py 导入模型
# ============================================================

from model import LSTMSettlementModel


# ============================================================
# 1. 基本配置
# ============================================================

# 数据文件
DATA_PATH = "./tower_settlement_data.csv"

# 结果保存目录
SAVE_DIR = "./results"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 保存文件
# ============================================================

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

HISTORY_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "training_history.csv"
)


# ============================================================
# 3. 预测参数
# ============================================================

# 使用当前时刻的信息
# 预测未来15个时间步
PRED_LEN = 15


# ============================================================
# 4. 模型参数
# ============================================================

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.2


# ============================================================
# 5. 训练参数
# ============================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

# Early Stopping
# 如果连续15个epoch验证集没有提升，则停止
PATIENCE = 15


# ============================================================
# 6. 数据集划分
# ============================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# 7. 随机种子
# ============================================================

SEED = 42


# ============================================================
# 8. 设备
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 9. 固定随机种子
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
# 10. Dataset
# ============================================================

class SettlementDataset(Dataset):
    """
    数据结构：

    当前时刻 t：

        SlopeHeight
        SlopeAngle
        Settlement_t

    未来15个时间步：

        Rain[t+1:t+15]
        Evap[t+1:t+15]

    预测：

        Settlement[t+1:t+15]


    static_input:
        [batch, 3]

        Height
        Angle
        Settlement_t


    future_input:
        [batch, 15, 2]

        Rain
        Evap


    target:
        [batch, 15, 1]

        Settlement
    """

    def __init__(
        self,
        data,
        pred_len=15
    ):

        self.pred_len = pred_len

        self.samples = []

        # ----------------------------------------------------
        # 按坡高 + 坡角分组
        # ----------------------------------------------------

        grouped = data.groupby(
            [
                "SlopeHeight_m",
                "SlopeAngle_deg"
            ],
            sort=False
        )

        # ----------------------------------------------------
        # 每个工况单独建立滑动窗口
        # ----------------------------------------------------

        for _, group in grouped:

            # 保持原始时间顺序
            group = group.reset_index(drop=True)

            # ------------------------------------------------
            # t -> t+15
            # ------------------------------------------------

            for i in range(
                len(group) - pred_len
            ):

                self.samples.append(
                    (
                        group,
                        i
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        group, i = self.samples[idx]

        # ====================================================
        # 当前时刻输入
        # ====================================================

        static_input = group.loc[
            i,
            [
                "SlopeHeight_m",
                "SlopeAngle_deg",
                "Settlement_mm"
            ]
        ].values.astype(
            np.float32
        )

        # ====================================================
        # 未来15个时间步
        # ====================================================

        future_data = group.iloc[
            i + 1:
            i + 1 + self.pred_len
        ]

        future_input = future_data[
            [
                "Rain_mm",
                "Evap_mm"
            ]
        ].values.astype(
            np.float32
        )

        # ====================================================
        # 未来15个时间步的真实沉降
        # ====================================================

        target = future_data[
            [
                "Settlement_mm"
            ]
        ].values.astype(
            np.float32
        )

        # ====================================================
        # 转Tensor
        # ====================================================

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
# 11. 读取数据
# ============================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

data = pd.read_csv(
    DATA_PATH
)

print(
    "Data shape:",
    data.shape
)

print(
    "Columns:",
    data.columns.tolist()
)


# ============================================================
# 12. 检查必要字段
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


# ============================================================
# 13. 检查缺失值
# ============================================================

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
# 14. 检查16个坡面工况
# ============================================================

cases = data[
    [
        "SlopeHeight_m",
        "SlopeAngle_deg"
    ]
].drop_duplicates()

print(
    "\nNumber of slope cases:",
    len(cases)
)

print(cases)

if len(cases) != 16:

    print(
        "\nWarning:"
    )

    print(
        "Expected 16 slope cases, "
        "but found",
        len(cases)
    )


# ============================================================
# 15. 按工况进行 Train / Validation / Test 划分
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

    # --------------------------------------------------------
    # 保持该工况自己的时间顺序
    # --------------------------------------------------------

    group = group.reset_index(
        drop=True
    )

    n = len(group)

    # --------------------------------------------------------
    # 检查数据量
    # --------------------------------------------------------

    if n <= PRED_LEN:

        raise ValueError(
            f"Case Height={height}, "
            f"Angle={angle} has only {n} rows."
        )

    # --------------------------------------------------------
    # 时间顺序划分
    # --------------------------------------------------------

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (
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

    # --------------------------------------------------------
    # 检查三个数据集是否足够构造15步预测
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

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


# ============================================================
# 16. 合并三个数据集
# ============================================================

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
# 17. 创建Scaler
# ============================================================

# ------------------------------------------------------------
# 17.1 Height + Angle
# ------------------------------------------------------------

static_scaler = MinMaxScaler()

static_scaler.fit(
    train_data[
        [
            "SlopeHeight_m",
            "SlopeAngle_deg"
        ]
    ]
)


# ------------------------------------------------------------
# 17.2 Settlement
# ------------------------------------------------------------

settlement_scaler = MinMaxScaler()

settlement_scaler.fit(
    train_data[
        [
            "Settlement_mm"
        ]
    ]
)


# ------------------------------------------------------------
# 17.3 Rain + Evap
# ------------------------------------------------------------

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
# 18. 数据标准化函数
# ============================================================

def scale_dataframe(
    data,
    static_scaler,
    settlement_scaler,
    future_scaler
):

    data = data.copy()

    # --------------------------------------------------------
    # Height + Angle
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Settlement
    # --------------------------------------------------------

    data[
        [
            "Settlement_mm"
        ]
    ] = settlement_scaler.transform(
        data[
            [
                "Settlement_mm"
            ]
        ]
    )

    # --------------------------------------------------------
    # Rain + Evap
    # --------------------------------------------------------

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


# ============================================================
# 19. 标准化 Train / Val / Test
# ============================================================

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
# 20. 创建Dataset
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
# 21. 创建DataLoader
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
# 22. 检查数据维度
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
# 23. 创建模型
# ============================================================

model = LSTMSettlementModel(
    static_size=3,
    future_size=2,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pred_len=PRED_LEN
)

model = model.to(
    DEVICE
)


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
# 24. Loss
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# 25. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 26. 单轮训练函数
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

        # ----------------------------------------------------
        # 放到GPU / CPU
        # ----------------------------------------------------

        static_input = static_input.to(
            device
        )

        future_input = future_input.to(
            device
        )

        target = target.to(
            device
        )

        # ----------------------------------------------------
        # 前向传播
        # ----------------------------------------------------

        prediction = model(
            static_input,
            future_input
        )

        # ----------------------------------------------------
        # 计算Loss
        # ----------------------------------------------------

        loss = criterion(
            prediction,
            target
        )

        # ----------------------------------------------------
        # 梯度清零
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # 反向传播
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # 更新参数
        # ----------------------------------------------------

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
# 27. 验证函数
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

            static_input = static_input.to(
                device
            )

            future_input = future_input.to(
                device
            )

            target = target.to(
                device
            )

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
# 28. 开始训练
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

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    # --------------------------------------------------------
    # 保存训练历史
    # --------------------------------------------------------

    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f}"
    )

    # --------------------------------------------------------
    # 验证集提升
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        print(
            f"Validation loss has not improved "
            f"for {PATIENCE} epochs."
        )

        break


# ============================================================
# 29. 保存训练历史
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
# 30. 保存Scaler
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
# 31. 加载最佳模型
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
# 32. 测试集预测
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


# ============================================================
# 33. 合并预测结果
# ============================================================

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
# 34. 反归一化
# ============================================================

prediction_shape = all_predictions.shape

target_shape = all_targets.shape


pred_original = settlement_scaler.inverse_transform(
    all_predictions.reshape(
        -1,
        1
    )
).reshape(
    prediction_shape
)


target_original = settlement_scaler.inverse_transform(
    all_targets.reshape(
        -1,
        1
    )
).reshape(
    target_shape
)


# ============================================================
# 35. 计算测试指标
# ============================================================

y_pred = pred_original.reshape(
    -1
)

y_true = target_original.reshape(
    -1
)


mse = mean_squared_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mse
)

mae = mean_absolute_error(
    y_true,
    y_pred
)

r2 = r2_score(
    y_true,
    y_pred
)


# ============================================================
# 36. 输出测试结果
# ============================================================

print("\n")
print("=" * 60)
print("Test Results")
print("=" * 60)

print(
    f"MSE  = {mse:.6f}"
)

print(
    f"RMSE = {rmse:.6f} mm"
)

print(
    f"MAE  = {mae:.6f} mm"
)

print(
    f"R²   = {r2:.6f}"
)


# ============================================================
# 37. 保存测试预测结果
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

        for step in range(PRED_LEN):

            prediction_records.append(
                {
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
# 38. 最终输出
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
    "\nTest prediction:"
)

print(
    PREDICTION_SAVE_PATH
)

print(
    "\nTraining history:"
)

print(
    HISTORY_SAVE_PATH
)

print("=" * 60)