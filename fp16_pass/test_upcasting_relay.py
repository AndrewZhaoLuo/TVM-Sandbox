"""
This file tests how upcasting works in TVM
"""
import tvm
from tvm import relay
import numpy as np


def run_module(mod, mod_params, target="llvm"):
    dev = tvm.device(target, 0)
    intrp = relay.create_executor("debug", mod, device=dev, target=target)
    result = intrp.evaluate()(**mod_params)
    if isinstance(result, tvm.runtime.container.ADT):
        result = [r.asnumpy() for r in result]
        return result
    else:
        return [result.asnumpy()]


def case_binop(binop=relay.add, lhs_dtype="float32", rhs_dtype="float32"):
    lhs = relay.var("lhs", shape=[3], dtype=lhs_dtype)
    rhs = relay.var("rhs", shape=[3], dtype=rhs_dtype)
    output = binop(lhs, rhs)

    return tvm.IRModule.from_expr(output), {
        "lhs": np.random.uniform(-128, 128, size=[3]).astype(lhs_dtype),
        "rhs": np.random.uniform(-128, 128, size=[3]).astype(rhs_dtype),
    }

def run_and_print(binop=relay.add, lhs_dtype="float32", rhs_dtype="float32"):
    mod, mod_params = case_binop(binop=binop, lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype)
    results = run_module(mod, mod_params)
    print(f"{binop}({mod_params['lhs']}, {mod_params['rhs']}) = {results}")

if __name__ == "__main__":
    # Integer + float upcasting works
    run_and_print(lhs_dtype='int8')
    run_and_print(lhs_dtype='int16')
    run_and_print(lhs_dtype='int32')
    run_and_print(lhs_dtype='int64')