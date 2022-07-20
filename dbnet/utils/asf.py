from mindspore import Tensor, context
import mindspore.nn as nn
import mindspore.ops as ops


class SpatialAttention(nn.Cell):
    def __init__(self, inner_channels, N=4):
        super(SpatialAttention, self).__init__()

        self.channel_wise = nn.SequentialCell([
            nn.Conv2d(inner_channels, inner_channels // N, 1),
            nn.Conv2d(inner_channels // N, inner_channels, 1)])
        self.spatial_wise = nn.SequentialCell([
            nn.Conv2d(1, 1, 3),
            nn.Conv2d(1, 1, 1)])
        self.attention_wise = nn.Conv2d(inner_channels, N, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.reduce_mean = ops.ReduceMean(True)
        self.split = ops.Split(1, N)
        self.stack = ops.Stack(0)

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
        y = self.stack(self.split(y))

        return y


class ASF(nn.Cell):
    def __init__(self, inner_channels, N=4):
        super(ASF, self).__init__()
        self.N = N

        self.conv = nn.Conv2d(inner_channels, inner_channels, 3, has_bias=True)
        self.spatial_attention = SpatialAttention(inner_channels, N)

        self.concat_1 = ops.Concat(1)
        self.concat_2 = ops.Concat(2)
        self.stack = ops.Stack(0)
        self.split = ops.Split(0, N)

    def construct(self, x: list):
        X = self.stack(x)

        S = self.conv(self.concat_1(x))
        A = self.spatial_attention(S)
        F = self.concat_2(self.split(A * X))

        return F.squeeze(0)


if __name__ == '__main__':
    context.set_context(device_id=5, mode=context.GRAPH_MODE)

    std = ops.StandardNormal()
    split = ops.Split(output_num=4)
    concat = ops.Concat()

    X = std((4, 16, 64, 50, 100))   # 特征图N, mini-batch, ...
    X = split(X)
    X = list(X)
    for i in range(len(X)):
        X[i] = X[i].squeeze()

    asf = ASF(256)
    result = asf(X)
    print(result.shape)
