# Printing and making sense of tvmscript and convs in CUDA
import tvm.topi.nn as nn_te
import tvm.te as te
from tvm import IRModule

from tvm import tir as T

from tvm.script import tir as T

if __name__ == "__main__":
    image = te.placeholder((1, 3, 1200, 1200), dtype="float32", name="image")
    filter = te.placeholder((64, 3, 7, 7), dtype="float32", name="filter")

    # Get the basic conv2d compute
    output = nn_te.conv2d(
        image, filter, strides=2, padding=3, dilation=1, layout="NCHW"
    )
    func = te.create_prim_func([image, filter, output])

    # Prim func is closer to C code
    print(func)

    ir_module_from_te = IRModule({"main": func})

    # TVMScript is more abstract, use above to reverse engineer some meanings
    print(ir_module_from_te.script())
