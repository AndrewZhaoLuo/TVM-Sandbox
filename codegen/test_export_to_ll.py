"""Exporting to an LLVM assembly language file (.ll)"""

import os

import tvm
from tvm import te

if __name__ == "__main__":
    TARGET = "llvm -opt-level=3 --fast-math"

    n = 128
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = tvm.te.create_schedule(C.op)
    m = tvm.lower(s, [A, B, C], name="test_add")
    rt_mod = tvm.build(m, target=TARGET)
    rt_mod.save("test.ll")
