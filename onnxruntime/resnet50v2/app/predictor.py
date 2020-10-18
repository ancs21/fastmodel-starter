import io
import os
import numpy as np
import requests as rq
import onnxruntime as ort
from PIL import Image
from app.utils import load_labels, softmax

class Predictor:
  def __init__(self, mode_path) -> None:
    self.model = ort.InferenceSession(f'{os.getcwd()}/{mode_path}', None)

  def pre_process(self, input_data):
    # Resize
    width = 224
    height = 224
    image = input_data.resize((width, height), Image.BILINEAR)

    # HWC -> CHW
    image = np.array(image).transpose(2, 0, 1)
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

  def predict(self, image_url):
    response = rq.get(image_url)
    image_bytes = io.BytesIO(response.content)
    image = Image.open(image_bytes)
    input_data = self.pre_process(image)
    input_name = self.model.get_inputs()[0].name  
    raw_result = self.model.run([], {input_name: input_data})
    res = self.post_process(raw_result)
    return res

  def post_process(self, raw_result):
    labels = load_labels(f'{os.getcwd()}/app/labels.json')
    res = softmax(np.array(raw_result)).tolist()
    idx = np.argmax(res)
    return labels[idx]


