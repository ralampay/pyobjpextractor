import sys
import argparse
import os
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.ss_obj_extractor import SsObjExtractor

KEY_Q = 113
KEY_J = 106
KEY_K = 107

def main():
  parser = argparse.ArgumentParser(description="PyObjPExtractor: Object Proposal Extractor program")

  parser.add_argument("--input-file", help="Input image file (jpg/jpeg only)", required=True, type=str)
  parser.add_argument("--output-dir", help="Output directory of saved image proposals", type=str, default="./")
  parser.add_argument("--ss-rect-increment", help="Increment value for SS", type=int, default=1)
  parser.add_argument("--ss-padding", help="Padding in pixels for drawing rectangles", type=int, default=5)

  args = parser.parse_args()

  input_file        = args.input_file
  output_dir        = args.output_dir
  ss_rect_increment = args.ss_rect_increment
  ss_padding        = args.ss_padding

  image = cv2.imread(input_file)

  extractor = SsObjExtractor(img=image, padding=ss_padding)

  extractor.exec()

  while True:
    cv2.imshow("Output", extractor.processed_img)
    
    # Record key press
    k = cv2.waitKey(0) & 0xFF

    # Press 'q' to quit
    if(k == KEY_Q):
      break

    # Press 'j' to increase number of rectangles
    elif(k == KEY_J):
      if(extractor.num_rects + ss_rect_increment > len(extractor.rects)):
        extractor.num_rects = len(extractor.rects)
      else:
        extractor.num_rects += ss_rect_increment
      print("Increasing regions to {}...".format(extractor.num_rects))

      extractor.draw_rectangles()

    # Press 'k' to decrease number of rectangles
    elif(k == KEY_K and extractor.num_rects > ss_rect_increment):
      extractor.num_rects -= ss_rect_increment
      extractor.draw_rectangles()
      print("Decreasing regions to {}...".format(extractor.num_rects))
      

  cv2.destroyAllWindows()

  print("Done.")

if __name__ == '__main__':
  main()
