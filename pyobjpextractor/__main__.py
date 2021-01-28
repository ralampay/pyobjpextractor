import sys
import argparse
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.ss_obj_extractor import SsObjExtractor
from lib.canny_obj_extractor import CannyObjExtractor

WINDOW_NAME="OUTPUT"

KEY_Q = 113
KEY_J = 106
KEY_K = 107

MODE_CHOICES = [
  "ss",
  "canny"
]

def mouse_callback(event, x, y, flags, param):
  print("Current Position (%d, %d)" % (x, y))

def main():
  parser = argparse.ArgumentParser(description="PyObjPExtractor: Object Proposal Extractor program")

  parser.add_argument("--mode", help="Mode of object proposal extractor", type=str, default="ss", choices=MODE_CHOICES)
  parser.add_argument("--input-file", help="Input image file (jpg/jpeg only)", required=True, type=str)
  parser.add_argument("--output-dir", help="Output directory of saved image proposals", type=str, default="./")
  parser.add_argument("--rect-increment", help="Increment value for num rects", type=int, default=1)
  parser.add_argument("--num-rects", help="Number of initial bounding boxes", type=int, default=500)
  parser.add_argument("--ss-padding", help="Padding in pixels for drawing rectangles", type=int, default=5)
  parser.add_argument("--ss-is-fast", help="Fast processing for SS", type=bool, default=True)

  args = parser.parse_args()

  mode            = args.mode
  input_file      = args.input_file
  output_dir      = args.output_dir
  rect_increment  = args.rect_increment
  num_rects       = args.num_rects
  ss_padding      = args.ss_padding
  ss_is_fast      = args.ss_is_fast

  image = cv2.imread(input_file)

  if mode == "ss":
    extractor = SsObjExtractor(img=image, padding=ss_padding, is_fast=ss_is_fast, num_rects=num_rects)
  elif mode == "canny":
    extractor = CannyObjExtractor(img=image, padding=ss_padding, num_rects=num_rects)

  extractor.exec()

  while True:
    cv2.imshow(WINDOW_NAME, extractor.processed_img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
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
