import torch
import torch.nn as nn


class LSTMSettlementModel(nn.Module):
    def __init__(
        self,
        static_size=3,      # Height, Angle, Settlement_t
        future_size=2,      # Rain, Evap
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        pred_len=15
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        # 当前状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 未来15天 Rain + Evap
        self.lstm = nn.LSTM(
            input_size=future_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出未来15天沉降
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, static_input, future_input):

        # static_input:
        # [batch, 3]
        # [Height, Angle, Settlement_t]

        # future_input:
        # [batch, 15, 2]
        # [Rain, Evap]

        # 当前状态编码
        state = self.state_encoder(static_input)

        # [batch, hidden_size]
        # → [num_layers, batch, hidden_size]
        h0 = state.unsqueeze(0).repeat(
            self.num_layers, 1, 1
        )

        # 初始 cell state
        c0 = torch.zeros_like(h0)

        # LSTM
        lstm_out, _ = self.lstm(
            future_input,
            (h0, c0)
        )

        # 输出
        output = self.fc(lstm_out)

        return output