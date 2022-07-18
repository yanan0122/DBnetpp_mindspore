from cv2 import split
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops


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
        self.split = ops.Split(0, N)

    def construct(self, x):
        x1 = self.channel_wise(x)

        x2 = self.reduce_mean(x1)
        x2 = self.spatial_wise(x2)
        x2 = self.relu(x2)
        x2 = self.spatial_wise(x2)
        x2 = self.sigmoid(x2)

        y = x1 + x2
        y = self.attention_wise(y)
        y = self.sigmoid(y)

        Y = self.expand_dims(y, 1)
        return self.split(Y)


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

    def construct(self, X):
        if len(X) != self.N:
            exit(1)

        S = self.conv(self.concat(X))
        A = self.spatial_attention(S)
        F = self.concat([ops.matmul(A[i], X[i]) for i in self.N])

        return F