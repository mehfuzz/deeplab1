import numpy as np
from PIL import Image
import io

import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
import streamlit as st
import tarfile
import os
import cv2
import time


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    target_dims=1920,1080
    resized_image1 = cv2.resize(seg_image, target_dims, interpolation 
=cv2.INTER_NEAREST)
    seg1=cv2.cvtColor(resized_image1,cv2.COLOR_RGB2BGR)
    #cv2.imwrite("87_seg.jpg",seg1)
    st.image(seg1,channels="BGR")
    return resized_image, seg_map

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

def label_to_color_image(label):
  
  colormap = create_cityscapes_label_colormap()
  return colormap[label]
  
def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""


 

  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #plt.imshow(seg_image)
  fig=plt.figure()
  plt.imshow(seg_image, alpha=0.3)
  st.pyplot(fig)
  plt.axis('off')
  #seg1=cv2.cvtColor(seg_image,cv2.COLOR_RGB2BGR)
  #resized_image = cv2.resize(segmentation_mask, target_dims, interpolation 
#=cv2.INTER_NEAREST)
  #cv2.imwrite("122_seg.jpg",seg1)





  plt.grid('off')
  plt.show()




if __name__ == '__main__':
  # Created comparing the colors for Cityscapes in 
  # deeplab/utils/get_dataset_colormap.py: https://github.com/tensorflow/models/blob/ea61bbf06c25068dd8f8e130668e36186187863b/research/deeplab/utils/get_dataset_colormap.py#L212
  # and the code in cityscapes: https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py

  st.title('Pretrained model demo')

  LABEL_NAMES = np.asarray([
      'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
      'bus', 'train', 'motorcycle', 'bycycle'])


  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


  #img_path = "87.png"
  model_path = "deeplabv3_cityscapes_train_2018_02_06.tar.gz"

  # load model
  model = DeepLabModel(model_path)

  # read image

  original_im = load_image()
  result = st.button('Run on image')
  if result:
        st.write('Calculating results...')
        start_time = time.time()
        resized_im, seg_map = model.run(original_im)
        ellapsed_time = time.time() - start_time
        print("Ellapsed time: " + str(ellapsed_time) + "s")
        vis_segmentation(resized_im, seg_map)
  # inferences DeepLab model
  
  
  
  # show inference result
  

