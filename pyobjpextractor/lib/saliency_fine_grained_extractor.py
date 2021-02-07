import cv2
import numpy as np

class SaliencyFineGrainedExtractor:
  def __init__(self, img=None, sigma=0.33, padding=5, sort_reverse=False, num_rects=100, min_area=2500, max_area=-1):
    self.img          = img
    self.rects        = []
    self.num_rects    = num_rects
    self.padding      = padding
    self.sigma        = sigma
    self.sort_reverse = sort_reverse
    self.saliency     = cv2.saliency.StaticSaliencyFineGrained_create()
    self.min_area     = min_area
    self.max_area     = max_area

  def exec(self):
    (self.success, self.saliency_map) = self.saliency.computeSaliency(self.img)
    self.saliency_map                 = (self.saliency_map * 255).astype("uint8")
    self.thresh_map                   = cv2.threshold(self.saliency_map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply Canny edge detection to the thresh_map
    self.median = np.median(self.thresh_map)
    self.lower  = int(max(0, (1.0 - self.sigma) * self.median))
    self.upper  = int(min(255, (1.0 + self.sigma) * self.median))
    self.edged  = cv2.Canny(self.thresh_map, self.lower, self.upper)

    self.rects = self.process()

    self.rects = sorted(self.rects, key=lambda x: x[2] * x[3], reverse=self.sort_reverse)

    # Filter only all objects >= self.rects
    if self.min_area > 0:
      temp = []
      for r in  self.rects:
        if r[2] * r[3] >= self.min_area:
          temp.append(r)

      self.rects = temp

    # Filter only all objects <= self.max_area
    if self.max_area > 0:
      temp = []
      for r in  self.rects:
        if r[2] * r[3] <= self.max_area:
          temp.append(r)

      self.rects = temp

    # Initialize num_rects
    if self.num_rects > len(self.rects):
      self.num_rects = len(self.rects)

    self.draw_rectangles()

  def draw_rectangles(self):
    # draw rectangles
    self.processed_img = self.img.copy()

    for i, rect in enumerate(self.rects):
      if(i < self.num_rects):
        x, y, w, h = rect
        cv2.rectangle(self.processed_img, (x - self.padding, y - self.padding), (x + w + self.padding, y + h + self.padding), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(self.processed_img, (x, y), (x + w, y + h), (255, 255, 0), 1, cv2.LINE_AA)

  def process(self):
    contours, hierarchy = cv2.findContours(self.thresh_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    rects = []

    for c in contours:
      epsilon = 0.1*cv2.arcLength(c, True)
      poly    = cv2.approxPolyDP(c, epsilon, True)

      rects.append(np.array(cv2.boundingRect(poly)))

    return rects
