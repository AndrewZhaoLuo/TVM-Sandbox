# Printing and making sense of tvmscript and convs in CUDA
import tvm
import tvm.te as te
import tvm.topi.cuda as cuda_topi
import tvm.topi.nn as nn_te
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

    # Schedule it
    output_sch = cuda_topi.schedule_conv2d_nchw(output)

    # Prim func is closer to C code
    func = te.create_prim_func([image, filter, output_sch])
    print(func)

    ir_module_from_te = IRModule({"main": func})

    # TVMScript is more abstract, use above to reverse engineer some meanings
    print(ir_module_from_te.script())
