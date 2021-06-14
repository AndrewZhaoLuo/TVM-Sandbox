# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import multiprocessing as mp
from os import path

import numpy as np
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.transform import AMPRewrite, InferType


def load_model(name, **kwargs):
    return tvmc.load(path.join("./models", name), **kwargs)


def get_one_conv_model(N=4, C_I=16, C_O=64, H=224, W=224, K=3, dtype="float32"):
    data_shape = [N, C_I, H, W]
    kernel_shape = [C_O, C_I, K, K]
    data = relay.var("data", dtype=dtype, shape=data_shape)
    weight = relay.var("weight", dtype=dtype, shape=kernel_shape)
    conv = relay.nn.conv2d(
        data=data,
        weight=weight,
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        kernel_size=K,
        channels=C_O,
    )
    mod = tvm.IRModule.from_expr(conv)
    params = {
        # "data": np.random.uniform(-1, 1, size=data_shape),
        "weight": np.random.uniform(-1, 1, size=kernel_shape).astype(dtype),
    }
    return tvmc.TVMCModel(mod, params)


def graph_optimize(tvmc_model, run_fp16_pass, run_other_opts):
    mod, params = tvmc_model.mod, tvmc_model.params
    # Weird functions we don't use are in there it's weird
    mod = tvm.IRModule.from_expr(mod["main"])

    if run_other_opts:
        # mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    if run_fp16_pass:
        mod = InferType()(mod)
        mod = AMPRewrite()(mod)

    if run_other_opts and run_fp16_pass:
        # run one more pass to clean up new subgraph
        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    return TVMCModel(mod, params)


def get_distillbert(run_pass=True, run_opts=True):
    tvmc_model = load_model("distilbert.onnx")
    return graph_optimize(tvmc_model, run_pass, run_opts)


def get_bert(run_pass=True, run_opts=True):
    tvmc_model = load_model("bert-base-uncased.pb")
    return graph_optimize(tvmc_model, run_pass, run_opts)


def get_yolo2(run_pass=True, run_opts=True):
    tvmc_model = load_model("yolov2-coco-9.onnx")
    return graph_optimize(tvmc_model, run_pass, run_opts)


def get_ssd_resnet(run_pass=True, run_opts=True):
    tvmc_model = load_model("ssd-10.onnx")
    return graph_optimize(tvmc_model, run_pass, run_opts)


if __name__ == "__main__":
    # macOS has 'spawn' as default which doesn't work for tvm
    mp.set_start_method("fork")

    for run_fp16_pass in [False, True]:
        print("FP16 pass" if run_fp16_pass else "FP32 pass")
        """Get Module"""
        # tvmc_model = get_one_conv_model(dtype="float16")
        # tvmc_model = get_one_conv_model(dtype="float32")
        # tvmc_model = get_distillbert()
        # tvmc_model = get_bert(run_pass=True)
        tvmc_model = get_yolo2(run_pass=run_fp16_pass)
        # tvmc_model = get_ssd_resnet(run_pass=run_fp16_pass)
        # tvmc_model.summary()

        # Create tuning artifacts
        target = "llvm -mcpu=apple-latest -mtriple=arm64-apple-macos"
        tuning_records = tvmc.tune(
            tvmc_model, target=target, trials=10000, repeat=5, tuner="xgb_knob"
        )

        # Create package artifacts
        package = tvmc.compile(tvmc_model, target=target, tuning_records=tuning_records)
        result = tvmc.run(package, device="cpu", repeat=1000, number=1)
        print(result)
        print()
