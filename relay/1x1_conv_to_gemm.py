import math
import pickle

import numpy as np
import tvm
from tvm import relay

# Testing transforming x,


def simple_example():
    # A single 1x1 conv
    c_in = 3
    c_out = 10
    dshape_nhwc = (10, 30, 18, c_in)
    kshape_hwio = (1, 1, c_in, c_out)
    oshape_nhwc = tuple(list(dshape_nhwc[:-1]) + [c_out])

    x_data = np.random.uniform(0, 1, size=dshape_nhwc).astype("float32")
    w_data = np.random.uniform(0, 1, size=kshape_hwio).astype("float32")

    def get_result_conv1x1():
        x = relay.var("x", shape=dshape_nhwc)
        w = relay.var("w", shape=kshape_hwio)

        # kshape and kernel_layout should have swapped IO.
        # kshape is HWOI and kernel_layout is HWIO
        y = relay.nn.conv2d(
            x, w, kernel_size=1, channels=10, data_layout="NHWC", kernel_layout="HWIO"
        )

        # We legalize since some schedules only support some layouts
        mod = tvm.IRModule.from_expr(y)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.Legalize()(mod)
        vm_exe = relay.create_executor("vm", mod=mod)
        return vm_exe.evaluate()(x_data, w_data).asnumpy()

    def get_result_matmul():
        # the equivalent as a batch_matmul, smash spatial dimensions
        x = relay.var("x", shape=dshape_nhwc)
        w = relay.var("w", shape=kshape_hwio)
        x = relay.reshape(x, [-1, c_in])
        w = relay.reshape(w, [c_in, c_out])

        y = relay.nn.matmul(x, w)
        y = relay.reshape(y, oshape_nhwc)

        mod = tvm.IRModule.from_expr(y)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.Legalize()(mod)
        vm_exe = relay.create_executor("vm", mod=mod)
        return vm_exe.evaluate()(x_data, w_data).asnumpy()

    result_conv1x1 = get_result_conv1x1()
    result_matmul = get_result_matmul()

    print("Matches?", (result_conv1x1 == result_matmul).all())


if __name__ == "__main__":
    simple_example()
