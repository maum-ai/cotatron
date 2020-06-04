import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum  # only used when track_running_stats=True
        if affine is True:
            raise NotImplementedError
        if track_running_stats is True:
            raise NotImplementedError

    def forward(self, x, lengths):
        # x: [N, C, L]
        # lengths: [N] torch.LongTensor
        lengths = lengths.view(-1, 1, 1).float()  # [N, 1, 1]
        sum_ = torch.sum(x, dim=2, keepdim=True)  # [N, C, 1]
        mean = sum_ / lengths  # [N, C, 1]
        sqsum = torch.sum(torch.pow(x, 2.0), dim=2, keepdim=True)  # [N, C, 1]
        sqmean = sqsum / lengths  # [N, C, 1]
        var = sqmean - torch.pow(mean, 2.0)  # [N, C, 1]

        return (x - mean) / torch.pow(var + self.eps, 0.5)


if __name__ == '__main__':
    instnorm = nn.InstanceNorm1d(1)
    p_instnorm = PaddedInstanceNorm1d(1)
    x = torch.tensor([-2., 1., 0., 3., 4.]).view(1, 1, -1)
    lengths = torch.LongTensor([5])
    
    print('-'*100)
    print('Check InstanceNorm1d == PaddedInstanceNorm1d')
    print('Input x: %s' % x)
    print('Input lengths: %s' % lengths)
    print('%s - nn.InstanceNorm1d(1)(x)' % instnorm(x))
    print('%s - PaddedInstanceNorm1d(1)(x, lengths)' % p_instnorm(x, lengths))
    
    print('-'*100)
    padded = torch.tensor([[-2., 1., 0., 3., 4., 0., 0.], [-2., 1., 0., 3., 4., 0., 0.]]).unsqueeze(1)
    padded_lengths = torch.LongTensor([5, 5])
    print('Input padded: %s, %s' % (padded, padded.shape))
    print('Input padded_lengths: %s' % padded_lengths)
    y = p_instnorm(padded, padded_lengths)
    print('%s - PaddedInstanceNorm1d(1)(x, lengths), %s' % (y, y.shape))
    print('-'*100)
    instnorm = nn.InstanceNorm1d(7, eps=1e-06)
    p_instnorm = PaddedInstanceNorm1d(7, eps=1e-06)
    x = torch.randn(3, 7, 11)
    lengths = torch.LongTensor([3, 9, 11])
    x[0, :, 3:] = 0.0
    x[1, :, 9:] = 0.0
    y0 = instnorm(x[0, :, :3].unsqueeze(0))
    y1 = instnorm(x[1, :, :9].unsqueeze(0))
    y2 = instnorm(x[2].unsqueeze(0))
    p_y = p_instnorm(x, lengths)
    print(y0 - p_y[0][:, :3] < 1e-6)
    print(y1 - p_y[1][:, :9] < 1e-6)
    print(y2 - p_y[2] < 1e-6)
    print(y.shape)
    print(p_y.shape)

