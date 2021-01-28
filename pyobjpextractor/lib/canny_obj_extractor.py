import cv2
import numpy as np

class CannyObjExtractor:
  def __init__(self, img=None, sigma=0.33, padding=5, sort_reverse=False, num_rects=100):
    self.img          = img
    self.sigma        = sigma
    self.padding      = padding
    self.sort_reverse = sort_reverse
    self.num_rects    = num_rects

  def exec(self):
    self.gray   = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
    self.median = np.median(self.gray)

    self.lower  = int(max(0, (1.0 - self.sigma) * self.median))
    self.upper  = int(min(255, (1.0 + self.sigma) * self.median))
    self.edged  = cv2.Canny(self.gray, self.lower, self.upper)

    self.rects = self.process()

    self.rects = sorted(self.rects, key=lambda x: x[2] * x[3], reverse=self.sort_reverse)

    # Initialize num_rects
    if self.num_rects > len(self.rects):
      self.num_rects = len(self.rects)

    self.draw_rectangles()

  def process(self):
    contours, hierarchy = cv2.findContours(self.edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    rects = []

    for c in contours:
      epsilon = 0.1*cv2.arcLength(c, True)
      poly    = cv2.approxPolyDP(c, epsilon, True)

      rects.append(cv2.boundingRect(poly))

    return rects

  def draw_rectangles(self):
    # draw rectangles
    self.processed_img = self.img.copy()

    for i, rect in enumerate(self.rects):
      if(i < self.num_rects):
        x, y, w, h = rect
        cv2.rectangle(self.processed_img, (x - self.padding, y - self.padding), (x + w + self.padding, y + h + self.padding), (0, 255, 0), 1, cv2.LINE_AA)
