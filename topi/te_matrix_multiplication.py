import tvm
import tvm.topi
from tvm import te

n = te.var("n")
m = te.var("m")
k = te.var("k")
A = te.placeholder((n, k), name="A")
B = te.placeholder((k, m), name="B")
k_reduce = te.reduce_axis((0, k), "k_axis")
C = te.compute(
    (n, m),
    lambda i, j: te.sum(A[i, k_reduce] * B[k_reduce, j], axis=k_reduce),
    name="C",
)
s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))