import torch
import torch.nn as nn
import torch.nn.functional as F

from .padded_instancenorm import PaddedInstanceNorm1d


# adopted GST's reference encoder.
# We changed:
# CNN stride = (2, 1) to preserve i/o time resolution, not (2, 2).
class ResidualEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.relu = nn.ReLU()
        self.stem = nn.Conv2d(
            1, hp.chn.residual[0], kernel_size=(7, 7), padding=(3, 3), stride=(2, 1))
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1))
            for in_channels, out_channels in zip(list(hp.chn.residual)[:-1], hp.chn.residual[1:])
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(channels) for channels in hp.chn.residual
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, T]
        self.fc = nn.Conv1d(hp.chn.residual[-1], hp.chn.residual_out, kernel_size=1)

        assert hp.ker.hann_window % 2 == 1, 'hp.ker.hann_window must be odd'
        hann_window = torch.hann_window(window_length=hp.ker.hann_window, periodic=False) 
        hann_window = hann_window.view(1, 1, -1) * (2.0 / (hp.ker.hann_window-1))
        self.register_buffer('hann', hann_window)
        self.padded_norm = PaddedInstanceNorm1d(hp.chn.residual_out)  # affine=False by default
        self.norm = nn.InstanceNorm1d(hp.chn.residual_out)  # affine=False by default

    def forward(self, mel, mask, lengths):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T]
        # mel: [B, mel, T]
        x = mel.unsqueeze(1)  # [B, 1, mel, T]
        x = self.stem(x)  # [B, chn.residual[0], T]

        for cnn, bn in zip(self.conv_layers, self.bn_layers):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            if mask is not None:
                x.masked_fill_(mask, 0.0)

        x = self.avgpool(x)  # [B, C, 1, T]
        x = x.squeeze(2)  # [B, C, T]
        x = self.fc(x)  # [B, chn.residual_out, T]

        if mask is not None:
            x.masked_fill_(mask.squeeze(1), 0.0)
            assert lengths is not None
            x = self.padded_norm(x, lengths)
            x.masked_fill_(mask.squeeze(1), 0.0)
        else:
            x = self.norm(x)

        x = torch.tanh(x)

        # smoothing with hann window
        x = x.view(-1, 1, x.size(2))  # [B*chn.residual_out, 1, T]
        x = F.conv1d(x, self.hann, padding=(self.hp.ker.hann_window-1)//2)
        x = x.view(-1, self.hp.chn.residual_out, x.size(2))  # [B, chn.residual_out, T]

        return x

    def inference(self, mel):
        return self.forward(mel, None, None)
