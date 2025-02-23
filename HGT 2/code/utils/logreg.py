import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.dropout = nn.Dropout(p=0.5)  # افزودن Dropout برای جلوگیری از overfitting
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # مقداردهی اولیه وزن‌ها با Xavier
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # مقداردهی اولیه بایاس‌ها با صفر

    def forward(self, seq):
        seq = self.dropout(seq)  # اعمال Dropout روی ورودی
        ret = self.fc(seq)
        return ret