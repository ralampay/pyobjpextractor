import cv2

class SsObjExtractor:
  def __init__(self, img=None, is_fast=True, num_rects=100, padding=5):
    self.img        = img
    self.is_fast    = is_fast
    self.rects      = []
    self.num_rects  = num_rects
    self.padding    = padding
    self.ss         = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    self.ss.setBaseImage(self.img)

    if self.is_fast:
      self.ss.switchToSelectiveSearchFast()
    else:
      self.ss.switchToSelectiveSearchQuality()

  def exec(self):
    self.rects = self.ss.process()

    self.draw_rectangles()

  def draw_rectangles(self):
    # draw rectangles
    self.processed_img = self.img.copy()

    for i, rect in enumerate(self.rects):
      if(i < self.num_rects):
        x, y, w, h = rect
        cv2.rectangle(self.processed_img, (x - self.padding, y - self.padding), (x + w + self.padding, y + h + self.padding), (0, 255, 0), 1, cv2.LINE_AA)
