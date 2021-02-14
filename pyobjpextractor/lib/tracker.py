import cv2
import numpy as np
import uuid
from tabulate import tabulate

class Tracker:
  def __init__(self, num_frames=10):
    self.num_frames = num_frames 
    self.data       = []

  def snapshot(self, extractor):
    d = {
      "frame":    extractor.img,
      "objects":  []
    }

    for r in extractor.rects:
      o = {
        "tl": (r[0], r[1]),
        "tr": (r[0] + r[2], r[1]),
        "bl": (r[0], r[1] + r[3]),
        "br": (r[0] + r[2], r[1] + r[3]),
        "xy": (int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)),
        "id": str(uuid.uuid1())
      }

      d["objects"].append(o)

    if len(self.data) == self.num_frames:
      self.data.pop()

    self.data.append(d)

  def print_data(self):
    for i in range(len(self.data)):
      print("Index: %d Num Objects: %d" % (i, len(self.data[i]["objects"])))

      row = []
      for o in self.data[i]["objects"]:
        row_data = [
          o["id"],
          o["tl"],
          o["tr"],
          o["bl"],
          o["br"],
          o["xy"]
        ]

        row.append(row_data)
      
      print(tabulate(row, ["ID", "Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"], tablefmt="grid"))
    print("")
