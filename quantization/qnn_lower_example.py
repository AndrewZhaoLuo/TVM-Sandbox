"""Example for lowering QNN into regular relay"""

from tvm import IRModule, relay

data = relay.var("data", dtype="int8", shape=[1, 1, 5, 5])
kernel = relay.var("kernel", dtype="int8", shape=[1, 1, 3, 3])
input_zero_point = relay.var("input_zero_point", dtype="int32", shape=[])
kernel_zero_point = relay.var("kernel_zero_point", dtype="int32", shape=[])
input_scale = relay.var("input_scale", dtype="float32", shape=[])
kernel_scale = relay.var("kernel_scale", dtype="float32", shape=[])

qnn = relay.qnn.op.conv2d(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    (3, 3),
    1,
)

mod = IRModule.from_expr(qnn)
mod = relay.transform.InferType()(mod)

# Lower qnn to relay
mod = relay.transform.Legalize(legalize_map_attr_name="FTVMQnnCanonicalize")(mod)

print(mod)

"""
Before Legalization:

def @main(%data: Tensor[(1, 1, 5, 5), int8], %kernel: Tensor[(1, 1, 3, 3), int8], %input_zero_point: int32, %kernel_zero_point: int32, %input_scale: float32, %kernel_scale: float32) {
  qnn.conv2d(%data, %kernel, %input_zero_point, %kernel_zero_point, %input_scale, %kernel_scale, padding=[0, 0, 0, 0], channels=1, kernel_size=[3, 3], out_dtype="int32")
}


After Legalization:

def @main(%data: Tensor[(1, 1, 5, 5), int8], %kernel: Tensor[(1, 1, 3, 3), int8], %input_zero_point: int32, %kernel_zero_point: int32, %input_scale: float32, %kernel_scale: float32) -> Tensor[(1, 1, 3, 3), int32] {
  %0 = reshape(%kernel_zero_point, int32, newshape=[-1]) /* ty=Tensor[(1), int32] */;
  %1 = cast(%data, Tensor[(1, 1, 5, 5), int8], dtype="int32") /* ty=Tensor[(1, 1, 5, 5), int32] */;
  %2 = sum(%1, Tensor[(1, 1, 5, 5), int32], axis=[1], keepdims=True) /* ty=Tensor[(1, 1, 5, 5), int32] */;
  %3 = multiply(%2, 9 /* ty=int32 */, Tensor[(1, 1, 5, 5), int32], int32) /* ty=Tensor[(1, 1, 5, 5), int32] */;
  %4 = nn.avg_pool2d(%3, Tensor[(1, 1, 5, 5), int32], pool_size=[3, 3]) /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %5 = expand_dims(%0, Tensor[(1), int32], axis=1, num_newaxis=2) /* ty=Tensor[(1, 1, 1), int32] */;
  %6 = repeat(%4, Tensor[(1, 1, 3, 3), int32], repeats=1, axis=1) /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %7 = nn.conv2d(%data, %kernel, Tensor[(1, 1, 5, 5), int8], Tensor[(1, 1, 3, 3), int8], padding=[0, 0, 0, 0], channels=1, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %8 = multiply(%5, %6, Tensor[(1, 1, 1), int32], Tensor[(1, 1, 3, 3), int32]) /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %9 = multiply(%input_zero_point, %5, int32, Tensor[(1, 1, 1), int32]) /* ty=Tensor[(1, 1, 1), int32] */;
  %10 = cast(%kernel, Tensor[(1, 1, 3, 3), int8], dtype="int32") /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %11 = sum(%10, Tensor[(1, 1, 3, 3), int32], axis=[1, 2, 3]) /* ty=Tensor[(1), int32] */;
  %12 = reshape(%11, Tensor[(1), int32], newshape=[1, 1, 1, 1]) /* ty=Tensor[(1, 1, 1, 1), int32] */;
  %13 = multiply(9 /* ty=int32 */, %9, int32, Tensor[(1, 1, 1), int32]) /* ty=Tensor[(1, 1, 1), int32] */;
  %14 = multiply(%input_zero_point, %12, int32, Tensor[(1, 1, 1, 1), int32]) /* ty=Tensor[(1, 1, 1, 1), int32] */;
  %15 = subtract(%7, %8, Tensor[(1, 1, 3, 3), int32], Tensor[(1, 1, 3, 3), int32]) /* ty=Tensor[(1, 1, 3, 3), int32] */;
  %16 = subtract(%13, %14, Tensor[(1, 1, 1), int32], Tensor[(1, 1, 1, 1), int32]) /* ty=Tensor[(1, 1, 1, 1), int32] */;
  add(%15, %16, Tensor[(1, 1, 3, 3), int32], Tensor[(1, 1, 1, 1), int32]) /* ty=Tensor[(1, 1, 3, 3), int32] */
}
"""
