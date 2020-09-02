import numpy as np
import imageio
import torch
import onnxruntime

ort_session = onnxruntime.InferenceSession("edsr_E.onnx")

lr = imageio.imread("001.png")
lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
lr = lr.astype(np.float32)
lr = lr[np.newaxis, :]

ort_inputs = {ort_session.get_inputs()[0].name: lr}
ort_outs = ort_session.run(None, ort_inputs)
img_out = ort_outs[0]
img_out = np.ascontiguousarray(img_out[0, :].transpose((1, 2, 0)))

imageio.imwrite("result.png", img_out)