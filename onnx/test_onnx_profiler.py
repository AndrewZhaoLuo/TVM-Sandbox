"""Example of using profiler in onnxrt"""

import numpy as np
import onnxruntime as ort

model_path = "models/resnet18-v1-7.onnx"
so = ort.SessionOptions()
so.log_severity_level = 3
so.enable_profiling = True
so.profile_file_prefix = "profiling_data"
session = ort.InferenceSession(model_path, sess_options=so)

result = session.run(
    None, {"data": np.random.uniform(0, 1, size=(1, 3, 224, 224)).astype("float32")}
)

# outputs a file like profiling_data_2022-02-18_12-53-56.json
