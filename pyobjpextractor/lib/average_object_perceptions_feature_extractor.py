import cv2
import numpy as np
from lib.utils import cv2_to_tensor

class AverageObjectPerceptionsFeatureExtractor:
  def __init__(self, cnn_autoencoder, img, rects, padding, img_width, img_height):
    self.cnn_autoencoder  = cnn_autoencoder
    self.img              = img
    self.rects            = rects
    self.padding          = padding
    self.num_objects      = len(self.rects)
    self.img_width        = img_width
    self.img_height       = img_height

  def execute(self):
    features = []

    images = self.rects_to_images()

    if len(images) > 0:
      result  = self.cnn_autoencoder.flatten(
                  cv2_to_tensor(images)
                )

      features = np.mean(result.detach().numpy(), axis=0)

      return features

    return features

  def rects_to_images(self):
    images = []

    buff_img = self.img.copy()

    for r in self.rects:
      x, y, w, h = r

      roi = buff_img[y:y+h, x:x+w]
      roi = cv2.resize(roi, (self.img_width, self.img_height))
      roi = roi / 255

      # Normalize the images
      images.append(roi)

    return images
