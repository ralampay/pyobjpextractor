import cv2
import numpy as np

class ExtractorUtil:
  def __init__(self, extractor):
    self.extractor = extractor

  def fetch_rects_in_bounds(self, pos_x, pos_y):
    rects = []

    for i, r in enumerate(self.extractor.rects):
      if(i < self.extractor.num_rects):
        if r[0] < pos_x < (r[0] + r[2]) and r[1] < pos_y < (r[1] + r[3]):
          # adjust padding
          r[0] = r[0] - self.extractor.padding
          r[1] = r[1] - self.extractor.padding
          r[2] = r[2] + self.extractor.padding
          r[3] = r[3] + self.extractor.padding
          rects.append(r)

    return rects
