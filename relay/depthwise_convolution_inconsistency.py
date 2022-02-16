"""
Meant to show inconsistency in depthwise convolutions. 
Each has different kernel shape representations depending on very subtle changes to type rel.
"""
import tvm
from tvm import IRModule, relay


def part1():
    # Kernel shape is [output_channels / multiplier, multiplier,  ...]
    data = relay.var("data", dtype="int8", shape=[1, 10, 5, 5])
    kernel = relay.var("kernel", dtype="int8", shape=[10, 2, 3, 3])

    conv = relay.nn.conv2d(data, kernel, groups=10, channels=20, kernel_size=(3, 3))

    mod = IRModule.from_expr(conv)
    mod = relay.transform.InferType()(mod)

    """
    Also has output: 
    For x86 target, depthwise_conv2d with channel multiplier greater than 1 is not optimized
    For x86 target, depthwise_conv2d with channel multiplier greater than 1 is not optimized
    """
    print(mod)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")


def part2():
    # Kernel shape is [output_channels, 1,  ...]
    data = relay.var("data", dtype="int8", shape=[1, 10, 5, 5])
    kernel = relay.var("kernel", dtype="int8", shape=[20, 1, 3, 3])

    conv = relay.nn.conv2d(data, kernel, groups=10, channels=20)

    mod = IRModule.from_expr(conv)
    mod = relay.transform.InferType()(mod)

    print(mod)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")


part1()
part2()
