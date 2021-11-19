from tvm import relay
import tvm
import numpy as np

# Test issues with cuda + batch_matmul + integer matmul. Improper support for mixed precision it seems

A = relay.var("A", shape=[5, 6, 4], dtype="int16")
B = relay.var("B", shape=[5, 4, 6], dtype="int16")

A_np = np.random.uniform(-10, 10, size=[5, 6, 4]).astype("int16")
B_np = np.random.uniform(-10, 10, size=[5, 4, 6]).astype("int16")

# Change out_dtype=int16 for this to work
result = relay.nn.batch_matmul(A, B, transpose_b=False, out_dtype="int32")

mod = tvm.IRModule.from_expr(result)
mod = relay.transform.InferType()(mod)
mod = relay.transform.Legalize()(mod)
vm_exe = relay.create_executor("debug", mod=mod, target="cuda")
result = vm_exe.evaluate()(A_np, B_np).asnumpy()

print(result)