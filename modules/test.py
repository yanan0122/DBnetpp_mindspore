import numpy as np

from mindspore import Tensor, context, nn, ops
import mindspore as ms
from mindspore import Tensor, nn, ops, context

class Foo(nn.Cell):

    def __init__(self):

        super(foo, self).__init__()

    def construct(self, x):

        return x*2

if __name__ == '__main__':
	
	context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=6)

	x = Tensor([1,2])
	foo = Foo()
	y = foo()
	print(y)