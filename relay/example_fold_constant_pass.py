"""Example of using passes in relay: CSE in this case"""
import tvm
from tvm import relay

input = relay.var("input", shape=[])
a = relay.const(10, dtype="float32")
b = relay.const(11, dtype="float32")
c = (a * b) + relay.const(1, dtype="float32")
d = (a * b) - relay.const(2, dtype="float32")
out = c + d + input
mod = tvm.IRModule.from_expr(out)
print(mod)

"""
Before:
def @main(%input: float32) {
  %0 = multiply(10f, 11f);
  %1 = multiply(10f, 11f);
  %2 = add(%0, 1f);
  %3 = subtract(%1, 2f);
  %4 = add(%2, %3);
  add(%4, %input)
}
"""

# Use Sequential to solve for pre-requisite passes
passes = tvm.transform.Sequential([relay.transform.FoldConstant()])
mod = passes(mod)
print(mod)
"""
After:
def @main(%input: float32) -> float32 {
  add(219f /* ty=float32 */, %input) /* ty=float32 */
}
"""
