import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import Tensor, context
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

        return y


class ASF(nn.Cell):
    def __init__(self, inner_channels=256, N=4):
        super(ASF, self).__init__()

        self.N = N
        self.conv = nn.Conv2d(inner_channels, inner_channels, 3, has_bias=True)
        self.spatial_attention = SpatialAttention(inner_channels, self.N)

        self.concat = ops.Concat()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        X = x.reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])

        S = self.conv(X)
        A = self.spatial_attention(S).transpose(1, 0, 2, 3)
        F = (A * x).reshape(1, x.shape[0]*x.shape[1], *x.shape[2:])

        return F


if __name__ == '__main__':
    std = ops.StandardNormal()
    split = ops.Split(output_num=4)
    concat = ops.Concat()

    X = std((4, 64, 50, 100))

    asf = ASF()
    result = asf(X)
    print(result.shape)
