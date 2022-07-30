from onnxruntime.quantization import quantize_dynamic
from onnxruntime.transformers import optimizer

model_fp32 = "models/arcfaceresnet100-8.onnx"
model_quant = "models/quant.onnx"
model_fp16 = "models/fp16.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant)

optimized_model = optimizer.optimize_model(model_fp32)
optimized_model.convert_model_float32_to_float16()
optimized_model.save_model_to_file(model_fp16)
