"""
How to list all ops registered so far.
"""

import tvm

op_names = tvm.ir.Op.list_op_names()

for name in op_names:
    print(name)
