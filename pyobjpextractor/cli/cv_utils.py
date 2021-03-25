import cv2
import uuid

def mouse_callback(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONUP:
    print("Current Position (%d, %d)" % (x, y))

    rects = param['extractor_util'].fetch_rects_in_bounds(x, y)

    print("Found {} regions!".format(len(rects)))

    regions = []

    for r in rects:
      roi = param['image'][r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
      filename = "{}/{}.jpg".format(param['output_dir'], str(uuid.uuid1()))

      print("Saving to {}".format(filename))

      # save to output dir
      cv2.imwrite(filename, roi)

