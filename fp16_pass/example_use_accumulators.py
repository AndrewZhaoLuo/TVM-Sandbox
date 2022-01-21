"""
Example of using AMP but using fp32 accumulators for nn.dense and nn.conv2d, overwritting default settings 
"""

import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.relay.op import register_mixed_precision_conversion

# Source: https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet18-v1-7.onnx
ONNX_MODEL = "models/resnet18-v1-7.onnx"

# Pick a priority > 10 to overwrite defaults, higher priorities take precedence
@register_mixed_precision_conversion("nn.conv2d", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    return [
        # always do main calculation in mixed_precision_type
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        # the dtype for the accumulator
        "float32",
        # the output dtype for the operation (usually fp16)
        mixed_precision_type,
    ]


@register_mixed_precision_conversion("nn.dense", level=11)
def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
    return [
        # always do main calculation in mixed_precision_type
        relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
        # the dtype for the accumulator
        "float32",
        # the output dtype for the operation (usually fp16)
        mixed_precision_type,
    ]


if __name__ == "__main__":
    onnx_model = onnx.load(ONNX_MODEL)
    mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ToMixedPrecision()(mod)

    # run one more pass to clean up new subgraph
    mod = relay.transform.EliminateCommonSubexpr()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.CombineParallelBatchMatmul()(mod)
    mod = relay.transform.FoldConstant()(mod)

    print(mod)

    intrp = relay.create_executor("debug", mod, target="llvm")
    np.random.seed(42)
    dummy_input = np.random.uniform(0, 1, size=(1, 3, 224, 224)).astype("float32")
    result_np = intrp.evaluate()(dummy_input)
    print(result_np)

    """
    Printing mod:

    Before registering ops:
    %82 = nn.conv2d(%81, meta[relay.Constant][95] /* ty=Tensor[(512, 512, 3, 3), float16] */, Tensor[(1, 512, 7, 7), float16], Tensor[(512, 512, 3, 3), float16], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="float16") /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %83 = nn.batch_norm(%82, meta[relay.Constant][96] /* ty=Tensor[(512), float16] */, meta[relay.Constant][97] /* ty=Tensor[(512), float16] */, meta[relay.Constant][98] /* ty=Tensor[(512), float16] */, meta[relay.Constant][99] /* ty=Tensor[(512), float16] */, Tensor[(1, 512, 7, 7), float16], Tensor[(512), float16], Tensor[(512), float16], Tensor[(512), float16], Tensor[(512), float16]) /* ty=(Tensor[(1, 512, 7, 7), float16], Tensor[(512), float16], Tensor[(512), float16]) */;
    %84 = %83.0;
    %85 = add(%77, %84, Tensor[(1, 512, 7, 7), float16], Tensor[(1, 512, 7, 7), float16]) /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %86 = nn.relu(%85, Tensor[(1, 512, 7, 7), float16]) /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %87 = cast(%86, Tensor[(1, 512, 7, 7), float16], dtype="float32") /* ty=Tensor[(1, 512, 7, 7), float32] */;
    %88 = nn.global_avg_pool2d(%87, Tensor[(1, 512, 7, 7), float32]) /* ty=Tensor[(1, 512, 1, 1), float32] */;
    %89 = nn.batch_flatten(%88, Tensor[(1, 512, 1, 1), float32]) /* ty=Tensor[(1, 512), float32] */;
    %90 = cast(%89, Tensor[(1, 512), float32], dtype="float16") /* ty=Tensor[(1, 512), float16] */;
    %91 = nn.dense(%90, meta[relay.Constant][100] /* ty=Tensor[(1000, 512), float16] */, Tensor[(1, 512), float16], Tensor[(1000, 512), float16], units=1000, out_dtype="float16") /* ty=Tensor[(1, 1000), float16] */;
    add(%91, meta[relay.Constant][101] /* ty=Tensor[(1000), float16] */, Tensor[(1, 1000), float16], Tensor[(1000), float16]) /* ty=Tensor[(1, 1000), float16] */

    After registering ops:
    %101 = nn.conv2d(%100, meta[relay.Constant][95] /* ty=Tensor[(512, 512, 3, 3), float16] */, Tensor[(1, 512, 7, 7), float16], Tensor[(512, 512, 3, 3), float16], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="float32") /* ty=Tensor[(1, 512, 7, 7), float32] */;
    %102 = cast(%101, Tensor[(1, 512, 7, 7), float32], dtype="float16") /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %103 = nn.batch_norm(%102, meta[relay.Constant][96] /* ty=Tensor[(512), float16] */, meta[relay.Constant][97] /* ty=Tensor[(512), float16] */, meta[relay.Constant][98] /* ty=Tensor[(512), float16] */, meta[relay.Constant][99] /* ty=Tensor[(512), float16] */, Tensor[(1, 512, 7, 7), float16], Tensor[(512), float16], Tensor[(512), float16], Tensor[(512), float16], Tensor[(512), float16]) /* ty=(Tensor[(1, 512, 7, 7), float16], Tensor[(512), float16], Tensor[(512), float16]) */;
    %104 = %103.0;
    %105 = add(%95, %104, Tensor[(1, 512, 7, 7), float16], Tensor[(1, 512, 7, 7), float16]) /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %106 = nn.relu(%105, Tensor[(1, 512, 7, 7), float16]) /* ty=Tensor[(1, 512, 7, 7), float16] */;
    %107 = cast(%106, Tensor[(1, 512, 7, 7), float16], dtype="float32") /* ty=Tensor[(1, 512, 7, 7), float32] */;
    %108 = nn.global_avg_pool2d(%107, Tensor[(1, 512, 7, 7), float32]) /* ty=Tensor[(1, 512, 1, 1), float32] */;
    %109 = nn.batch_flatten(%108, Tensor[(1, 512, 1, 1), float32]) /* ty=Tensor[(1, 512), float32] */;
    %110 = cast(%109, Tensor[(1, 512), float32], dtype="float16") /* ty=Tensor[(1, 512), float16] */;
    %111 = nn.dense(%110, meta[relay.Constant][100] /* ty=Tensor[(1000, 512), float16] */, Tensor[(1, 512), float16], Tensor[(1000, 512), float16], units=1000, out_dtype="float32") /* ty=Tensor[(1, 1000), float32] */;
    %112 = cast(%111, Tensor[(1, 1000), float32], dtype="float16") /* ty=Tensor[(1, 1000), float16] */;
    add(%112, meta[relay.Constant][101] /* ty=Tensor[(1000), float16] */, Tensor[(1, 1000), float16], Tensor[(1000), float16]) /* ty=Tensor[(1, 1000), float16] */
    """
