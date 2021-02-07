import sys
import argparse
import os
import cv2
import uuid
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.ss_obj_extractor import SsObjExtractor
from lib.canny_obj_extractor import CannyObjExtractor
from lib.extractor_util import ExtractorUtil
from lib.saliency_fine_grained_extractor import SaliencyFineGrainedExtractor

WINDOW_NAME="OUTPUT"

KEY_Q = 113
KEY_J = 106
KEY_K = 107

MODE_CHOICES = [
  "ss",
  "canny",
  "sfg"
]

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

def main():
  parser = argparse.ArgumentParser(description="PyObjPExtractor: Object Proposal Extractor program")

  parser.add_argument("--mode", help="Mode of object proposal extractor", type=str, default="ss", choices=MODE_CHOICES)
  parser.add_argument("--input-file", help="Input image file (jpg/jpeg only)", required=True, type=str)
  parser.add_argument("--output-dir", help="Output directory of saved image proposals", type=str, default="./")
  parser.add_argument("--rect-increment", help="Increment value for num rects", type=int, default=1)
  parser.add_argument("--num-rects", help="Number of initial bounding boxes", type=int, default=500)
  parser.add_argument("--padding", help="Padding in pixels for drawing rectangles", type=int, default=5)
  parser.add_argument("--ss-is-fast", help="Fast processing for SS", type=bool, default=True)
  parser.add_argument("--canny-sigma", help="sigma for auto edge calculation for Canny", type=float, default=0.33)

  args = parser.parse_args()

  mode            = args.mode
  input_file      = args.input_file
  output_dir      = args.output_dir
  rect_increment  = args.rect_increment
  num_rects       = args.num_rects
  padding         = args.padding
  ss_is_fast      = args.ss_is_fast
  canny_sigma     = args.canny_sigma

  image = cv2.imread(input_file)

  if mode == "ss":
    extractor = SsObjExtractor(img=image, padding=padding, is_fast=ss_is_fast, num_rects=num_rects)
  elif mode == "canny":
    extractor = CannyObjExtractor(img=image, padding=padding, num_rects=num_rects, sigma=canny_sigma)
  elif mode == "sfg":
    extractor = SaliencyFineGrainedExtractor(img=image, padding=padding, sigma=canny_sigma, num_rects=num_rects)

  extractor.exec()

  callback_params = {
    'extractor': extractor,
    'extractor_util': ExtractorUtil(extractor),
    'image': image,
    'output_dir': output_dir
  }

  while True:
    cv2.imshow(WINDOW_NAME, extractor.processed_img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, callback_params)
    
    # Record key press
    k = cv2.waitKey(0) & 0xFF

    # Press 'q' to quit
    if(k == KEY_Q):
      break

    # Press 'j' to increase number of rectangles
    elif(k == KEY_J):
      if(extractor.num_rects + rect_increment > len(extractor.rects)):
        extractor.num_rects = len(extractor.rects)
      else:
        extractor.num_rects += rect_increment
      print("Increasing regions to {}...".format(extractor.num_rects))

      extractor.draw_rectangles()

    # Press 'k' to decrease number of rectangles
    elif(k == KEY_K and extractor.num_rects > rect_increment):
      extractor.num_rects -= rect_increment
      extractor.draw_rectangles()
      print("Decreasing regions to {}...".format(extractor.num_rects))
      

  cv2.destroyAllWindows()

  print("Done.")

if __name__ == '__main__':
  main()
