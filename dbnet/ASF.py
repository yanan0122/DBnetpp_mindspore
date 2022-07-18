import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import context
from numpy import expand_dims
context.set_context(device_id=5, mode=context.GRAPH_MODE)

class SpatialAttention(nn.Cell):
    def __init__(self, inner_channels, N):
        super(SpatialAttention, self).__init__()

        self.channel_wise = nn.SequentialCell([
            nn.Conv2d(inner_channels, inner_channels // N, 1),
            nn.Conv2d(inner_channels // N, inner_channels, 1)
        ])
        self.spatial_wise = nn.SequentialCell([
            nn.Conv2d(1, 1, 3),
            nn.Conv2d(1, 1, 1)
        ])
        self.attention_wise = nn.Conv2d(inner_channels, N, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.reduce_mean = ops.ReduceMean(True)
        self.expand_dims = ops.ExpandDims()
        self.split = ops.Split(1, N)

    def construct(self, x):
        x1 = self.channel_wise(x)

        x2 = self.reduce_mean(x1, 1)
        x2 = self.spatial_wise(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_wise(x2)
        x2 = self.sigmoid(x2)

        y = x1 + x2
        y = self.attention_wise(y)
        y = self.sigmoid(y)

        return self.split(y)


class ASF(nn.Cell):
    '''
    The input for calling must be a list or tuple consisting of tensors.
    '''
    def __init__(self, inner_channels=256, N=4):
        super(ASF, self).__init__()

        self.N = N
        self.conv = nn.Conv2d(inner_channels, inner_channels, 3, has_bias=True)
        self.spatial_attention = SpatialAttention(inner_channels, self.N)

        self.concat = ops.Concat()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        if len(x) != self.N:
            exit(1)
        X = self.expand_dims(self.concat(x), 0)

        S = self.conv(X)
        A = self.spatial_attention(S)

        F = self.concat([(A[i] * x[i]).squeeze() for i in range(self.N)])
        return self.expand_dims(F, 0)


if __name__ == '__main__':
    std = ops.StandardNormal()
    split = ops.Split(output_num=4)
    concat = ops.Concat()

    X = std((4, 64, 50, 100))
    X = split(X)
    X = list(X)
    for i in range(len(X)):
        X[i] = X[i].squeeze()
    y = concat(X)

    asf = ASF()
    result = asf(X)
    print(result.shape)