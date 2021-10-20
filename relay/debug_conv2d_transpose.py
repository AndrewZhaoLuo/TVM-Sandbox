import math
import pickle

import numpy as np
import tvm
from tvm import relay

# Testing layout issues with conv2d_tranpose and oddities with some layouts being incompatible

dshape_nhwc = (1, 18, 18, 3)
kshape_hwoi = (3, 3, 10, 3)
oshape_nhwc = (1, 36, 36, 10)
x = relay.var("x", shape=dshape_nhwc)
w = relay.var("w", shape=kshape_hwoi)

# kshape and kernel_layout should have swapped IO.
# kshape is HWOI and kernel_layout is HWIO
y = relay.nn.conv2d_transpose(
    x,
    w,
    channels=10,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding=(1, 1),
    output_padding=(1, 1),
    data_layout="NHWC",
    kernel_layout="HWOI",
)

# We legalize since some schedules only support some layouts
mod = tvm.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
mod = relay.transform.Legalize()(mod)
vm_exe = relay.create_executor("vm", mod=mod)
result = vm_exe.evaluate()(
    np.random.uniform(0, 1, size=dshape_nhwc).astype("float32"),
    np.random.uniform(0, 1, size=kshape_hwoi).astype("float32"),
).asnumpy()
