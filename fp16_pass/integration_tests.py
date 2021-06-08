import tempfile
from collections import defaultdict
from os import path
from typing import *

import numpy as np
import onnx
import torch.onnx
import torchvision
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.relay.op.tensor import exp
from tvm.relay.testing import densenet, lstm, mobilenet, resnet, resnet_3d, squeezenet
from tvm.relay.transform import AMPRewrite
from tvm.relay.transform.transform import AnnotateSpans, InferType

MODELS_DIR = "./models/"


def run_module(mod, mod_params):
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("debug", mod, device=dev, target="llvm")
    result = intrp.evaluate()(**mod_params)
    if isinstance(result, tvm.runtime.container.ADT):
        result = [r.asnumpy() for r in result]
        return result
    else:
        return [result.asnumpy()]


def verify_fp32_fp16_output_close(mod, mod_params, rtol=1e-3, atol=0, run_opt=True):
    mod = InferType()(mod)
    result_fp32 = run_module(mod, mod_params)

    if run_opt:
        # mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    mod = AMPRewrite()(mod)

    if run_opt:
        # run one more pass to clean up new subgraph
        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    result_fp16 = run_module(mod, mod_params)

    # Ensure the results are close
    for fp32, fp16 in zip(result_fp32, result_fp16):
        np.testing.assert_allclose(fp32, fp16, rtol=rtol, atol=atol)

    return mod


# Native relay models
def test_resnet18():
    np.random.seed(4321)
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_resnet18_3d():
    np.random.seed(3215)
    mod, mod_params = resnet_3d.get_workload(
        1, 5, num_layers=18, image_shape=(1, 3, 32, 32)
    )
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 3, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_mobilenet():
    np.random.seed(4615)

    mod, mod_params = mobilenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_densenet():
    np.random.seed(3222)
    mod, mod_params = densenet.get_workload(
        classes=5, batch_size=1, image_shape=(1, 224, 224)
    )
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 224, 224)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_squeezenet():
    np.random.seed(5628)
    mod, mod_params = squeezenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")
    verify_fp32_fp16_output_close(mod, mod_params)


# Straight image classification models
def test_onnx_resnet18():
    model_path = path.join(MODELS_DIR, "resnet18-v1-7.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data"] = np.random.uniform(0, 1, size=[1, 3, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_efficientnet():
    model_path = path.join(MODELS_DIR, "efficientnet-lite4-11.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["images:0"] = np.random.uniform(0, 1, size=[1, 224, 224, 3]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_densenet():
    model_path = path.join(MODELS_DIR, "densenet-3.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data_0"] = np.random.uniform(0, 1, size=[1, 3, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_inceptionv3():
    model_path = path.join(MODELS_DIR, "inceptionv3.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input.1": [1, 3, 299, 299]}
    )
    mod_params["input.1"] = np.random.uniform(0, 1, size=[1, 3, 299, 299]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Object detection models
def test_onnx_tinyyolo2():
    model_path = path.join(MODELS_DIR, "tinyyolov2-7.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"image": [1, 3, 416, 416]}
    )
    mod_params["image"] = np.random.uniform(0, 1, size=[1, 3, 416, 416]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_yolo2():
    model_path = path.join(MODELS_DIR, "yolov2-coco-9.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input.1": [1, 3, 416, 416]}
    )
    mod_params["input.1"] = np.random.uniform(0, 1, size=[1, 3, 416, 416]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Face recognition / embedding
def test_onnx_arcfaceresnet():
    model_path = path.join(MODELS_DIR, "arcfaceresnet100-8.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data"] = np.random.uniform(0, 1, size=[1, 3, 112, 112]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_rfb():
    model_path = path.join(MODELS_DIR, "version-RFB-320.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["input"] = np.random.uniform(0, 1, size=[1, 3, 240, 320]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Super resolution
def test_onnx_superresolution():
    model_path = path.join(MODELS_DIR, "super-resolution-10.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input": [1, 1, 224, 224]}
    )
    mod_params["input"] = np.random.uniform(0, 1, size=[1, 1, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# NLP models (ruh roh!)
def test_onnx_gpt2():
    model_path = path.join(MODELS_DIR, "gpt2-10.onnx")
    onnx_model = onnx.load(model_path)

    mod, mod_params = relay.frontend.from_onnx(onnx_model, shape={"input1": [1, 1, 1]})
    mod_params["input1"] = np.random.randint(0, 100, size=[1, 1, 1])
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_distillbert():
    model_path = path.join(MODELS_DIR, "distilbert.onnx")
    onnx_model = onnx.load(model_path)

    mod, mod_params = relay.frontend.from_onnx(onnx_model, shape={"input.1": [10, 100]})
    mod_params["input.1"] = np.random.randint(0, 100, size=[10, 100])
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_pb_bert():
    tvmc_model = tvmc.load(path.join(MODELS_DIR, "bert-base-uncased.pb"))
    mod, mod_params = tvmc_model.mod, tvmc_model.params
    # Weird functions we don't use are in there it's weird
    mod = tvm.IRModule.from_expr(mod["main"])
    mod_params["x"] = np.random.randint(0, 100, size=[1, 128]).astype("int32")
    output_mod = verify_fp32_fp16_output_close(
        mod, mod_params, atol=0.05, rtol=0.01, run_opt=True
    )
    assert not tvm.ir.structural_equal(mod, output_mod)
