"""Scratch pad for testing dropout"""

from numpy import random
from tvm import IRModule, relay

shape_input = [1, 3, 5, 5]
input_data = relay.var("input", shape=shape_input)
ratio = relay.var("ratio", shape=[])
result = relay.nn.dropout(input_data, ratio)

mod = IRModule.from_expr(result)
print(mod)

intrp = relay.create_executor("debug", mod, target="llvm")
result_np = intrp.evaluate()(
    input=random.uniform(-10, 10, shape=shape_input), ratio=0.5
)
