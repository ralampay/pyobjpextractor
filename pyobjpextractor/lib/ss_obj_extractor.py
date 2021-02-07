import cv2

class SsObjExtractor:
  def __init__(self, img=None, is_fast=True, num_rects=100, padding=5, sort_reverse=False, min_area=2500, max_area=-1):
    self.img          = img
    self.is_fast      = is_fast
    self.rects        = []
    self.num_rects    = num_rects
    self.padding      = padding
    self.sort_reverse = sort_reverse
    self.ss           = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    self.min_area     = min_area
    self.max_area     = max_area

    self.ss.setBaseImage(self.img)

    if self.is_fast:
      self.ss.switchToSelectiveSearchFast()
    else:
      self.ss.switchToSelectiveSearchQuality()

  def exec(self):
    self.rects = self.ss.process()

    self.rects = sorted(self.rects, key=lambda x: x[2] * x[3], reverse=self.sort_reverse)

    # Initialize num_rects
    if self.num_rects > len(self.rects):
      self.num_rects = len(self.rects)

    # Filter only all objects >= self.min_area
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

    self.draw_rectangles()

  def draw_rectangles(self):
    # draw rectangles
    self.processed_img = self.img.copy()

    for i, rect in enumerate(self.rects):
      if(i < self.num_rects):
        x, y, w, h = rect
        cv2.rectangle(self.processed_img, (x - self.padding, y - self.padding), (x + w + self.padding, y + h + self.padding), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(self.processed_img, (x, y), (x + w, y + h), (255, 255, 0), 1, cv2.LINE_AA)
