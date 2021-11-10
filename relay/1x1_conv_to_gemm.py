import math
import pickle

import numpy as np
import tvm
from numpy.core.numeric import Inf
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay.dataflow_pattern import DFPatternCallback, is_op, rewrite, wildcard
from tvm.relay.transform.transform import InferType

# Testing transforming conv1x1 NHWC --> matmul and using operator libraries or schedules.


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


class Callback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        self.x = wildcard()
        self.w = wildcard()
        self.conv = is_op("nn.conv2d")(self.x, self.w).has_attr(
            {
                "groups": 1,
                "data_layout": "NHWC",
                "kernel_layout": "HWIO",
                "kernel_size": [1, 1],
                "strides": [1, 1],
                "dilation": [1, 1],
            }
        )
        self.pattern = self.conv

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        w = node_map[self.w][0]

        x_shape = x.checked_type.shape
        w_shape = w.checked_type.shape

        o_shape = list(x_shape)
        o_shape[-1] = w_shape[-1]

        x = relay.reshape(x, [-1, x_shape[-1]])
        w = relay.reshape(w, w_shape[-2:])

        y = relay.nn.matmul(x, w)
        y = relay.reshape(y, o_shape)

        return y


def rewrite_example():
    # A graph of arithmetic operators that are functional equivalent to batch_norm.
    c_in = 3
    c_out = 10
    dshape_nhwc = (10, 30, 18, c_in)
    kshape_hwio = (1, 1, c_in, c_out)

    x_data = np.random.uniform(0, 1, size=dshape_nhwc).astype("float32")
    w_data = np.random.uniform(0, 1, size=kshape_hwio).astype("float32")

    x = relay.var("x", shape=dshape_nhwc)
    w = relay.var("w", shape=kshape_hwio)

    # kshape and kernel_layout should have swapped IO.
    # kshape is HWOI and kernel_layout is HWIO
    y = relay.nn.conv2d(
        x, w, kernel_size=1, channels=10, data_layout="NHWC", kernel_layout="HWIO"
    )

    mod_orig = IRModule.from_expr(y)
    mod_orig = InferType()(mod_orig)
    new_mod = rewrite(Callback(), mod_orig["main"].body)
    new_mod = IRModule.from_expr(new_mod)

    vm_exe = relay.create_executor("vm", mod=mod_orig)
    orig_result = vm_exe.evaluate()(x_data, w_data).asnumpy()

    vm_exe = relay.create_executor("vm", mod=new_mod)
    new_result = vm_exe.evaluate()(x_data, w_data).asnumpy()

    print("Matches?", (orig_result == new_result).all())


if __name__ == "__main__":
    simple_example()
    rewrite_example()
