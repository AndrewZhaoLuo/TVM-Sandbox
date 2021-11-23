import tflite2onnx

tflite_path = "models/tflite_model_zoo/mobilebert.tflite"
onnx_path = "models/mobilebert.onnx"

# Limited coverage, try tf2onnx instead
tflite2onnx.convert(tflite_path, onnx_path)