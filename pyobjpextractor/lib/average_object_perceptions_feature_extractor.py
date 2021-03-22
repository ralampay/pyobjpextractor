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
      image_data = []

      obj_predictions = self.cnn_autoencoder.predict(cv2_to_tensor(images))

      for i in range(len(obj_predictions)):
        if obj_predictions[i] == 1:
          image_data.append(images[i])    


      if len(image_data) > 0:
        result  = self.cnn_autoencoder.flatten(
                    cv2_to_tensor(image_data)
                  )


        x = result.detach().numpy()

        print("Found %d of %d objects!" % (len(x), len(images)))

        denominator = x.max(axis=0)

        if denominator.tolist().count(0) == 0:
          x = x / denominator

          features = np.sum(x, axis=0)

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
