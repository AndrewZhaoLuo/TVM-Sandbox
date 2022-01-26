"""
How to list all ops registered so far for quantization and in a generic registry.
"""

import tvm

op_names = tvm.ir.Op.list_op_names()

print("Ops:")
for name in sorted(op_names):
    print(name)
print()

print("Those with registered FTVMFakeQuantizationToInteger")
for name in sorted(op_names):
    op = tvm.ir.Op.get(name)
    if op.get_attr("FTVMFakeQuantizationToInteger"):
        print(name)
