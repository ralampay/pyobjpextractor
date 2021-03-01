import sys
import argparse
import os
import cv2
import uuid
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch import tensor
import torch

from lib.ss_obj_extractor import SsObjExtractor
from lib.canny_obj_extractor import CannyObjExtractor
from lib.extractor_util import ExtractorUtil
from lib.saliency_fine_grained_extractor import SaliencyFineGrainedExtractor
from lib.tracker import Tracker
from lib.cnn_autoencoder import CnnAutoencoder

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

  parser.add_argument("--video-index", help="Video index for capture", type=int, default=-1)
  parser.add_argument("--video-file", help="Video file for capture", type=str)
  parser.add_argument("--mode", help="Mode of object proposal extractor", type=str, default="ss", choices=MODE_CHOICES)
  parser.add_argument("--input-file", help="Input image file (jpg/jpeg only)", type=str)
  parser.add_argument("--output-dir", help="Output directory of saved image proposals", type=str, default="./")
  parser.add_argument("--rect-increment", help="Increment value for num rects", type=int, default=1)
  parser.add_argument("--num-rects", help="Number of initial bounding boxes", type=int, default=500)
  parser.add_argument("--min-area", help="Minimum area for proposed objects", type=int, default=625)
  parser.add_argument("--max-area", help="Maximum area for proposed objects", type=int, default=62500)
  parser.add_argument("--padding", help="Padding in pixels for drawing rectangles", type=int, default=5)
  parser.add_argument("--ss-is-fast", help="Fast processing for SS", type=bool, default=True)
  parser.add_argument("--canny-sigma", help="sigma for auto edge calculation for Canny", type=float, default=0.33)

  # CNN Parameters
  parser.add_argument("--cnn-a-model", help="Model file (pth) for CNN Autoencoder", type=str, default="./model.pth")
  parser.add_argument("--cnn-a-img-height", help="Image height for CNN Autoencoder model", type=int, default=100)
  parser.add_argument("--cnn-a-img-width", help="Image width for CNN Autoencoder model", type=int, default=100)
  parser.add_argument("--cnn-a-layers", help="Layers for CNN Autoencoder model", type=int, nargs='+')
  parser.add_argument("--cnn-a-num-channels", help="Number of channels for CNN Autoencoder model", type=int, default=3)
  parser.add_argument("--cnn-a-scale", help="Scale for CNN Autoencoder model", type=int, default=2)
  parser.add_argument("--cnn-a-padding", help="Padding for CNN Autoencoder model", type=int, default=1)
  parser.add_argument("--cnn-a-kernel-size", help="Kernel size for CNN Autoencoder model", type=int, default=3)

  args = parser.parse_args()

  mode            = args.mode
  input_file      = args.input_file
  output_dir      = args.output_dir
  rect_increment  = args.rect_increment
  num_rects       = args.num_rects
  padding         = args.padding
  ss_is_fast      = args.ss_is_fast
  canny_sigma     = args.canny_sigma
  min_area        = args.min_area
  max_area        = args.max_area
  video_index     = args.video_index
  video_file      = args.video_file

  # Device
  if torch.cuda.is_available():
    dev = "cuda:0"
    print("CUDA is available...")
  else:
    dev = "cpu"

  device = torch.device(dev)

  # CNN Parameters
  cnn_a_model         = args.cnn_a_model
  cnn_a_img_height    = args.cnn_a_img_height
  cnn_a_img_width     = args.cnn_a_img_width
  cnn_a_layers        = args.cnn_a_layers
  cnn_a_num_channels  = args.cnn_a_num_channels
  cnn_a_scale         = args.cnn_a_scale
  cnn_a_padding       = args.cnn_a_padding
  cnn_a_kernel_size   = args.cnn_a_kernel_size

  cnn_autoencoder = CnnAutoencoder(
                      scale=cnn_a_scale, 
                      channel_maps=cnn_a_layers, 
                      padding=cnn_a_padding, 
                      kernel_size=cnn_a_kernel_size,
                      num_channels=cnn_a_num_channels,
                      img_width=cnn_a_img_width,
                      img_height=cnn_a_img_height,
                      device=dev
                    )

  cnn_autoencoder.to(device)

  if video_file:
    # initialize video capture instance
    cap = cv2.VideoCapture(video_file)

    # initialize tracker for tracking
    tracker = Tracker()

    while True:
      ret, frame = cap.read()

      if mode == "ss":
        extractor = SsObjExtractor(img=frame, padding=padding, is_fast=ss_is_fast, num_rects=num_rects, min_area=min_area, max_area=max_area)
      elif mode == "canny":
        extractor = CannyObjExtractor(img=frame, padding=padding, num_rects=num_rects, sigma=canny_sigma, min_area=min_area, max_area=max_area)
      elif mode == "sfg":
        extractor = SaliencyFineGrainedExtractor(img=frame, padding=padding, sigma=canny_sigma, num_rects=num_rects, min_area=min_area, max_area=max_area)

      extractor.exec()

      callback_params = {
        'extractor': extractor,
        'extractor_util': ExtractorUtil(extractor),
        'image': frame,
        'output_dir': output_dir
      }

      # Save data in tracker
      tracker.snapshot(extractor)

      tracker.print_data()

      cv2.imshow(WINDOW_NAME, extractor.processed_img)
      cv2.setMouseCallback(WINDOW_NAME, mouse_callback, callback_params)
      
      # Record key press
      k = cv2.waitKey(25) & 0xFF

      # Press 'q' to quit
      if(k == KEY_Q):
        break
    
    cap.release()
  elif video_index >= 0:
    # initialize video capture instance
    cap = cv2.VideoCapture(video_index)

    # initialize tracker for tracking
    tracker = Tracker()

    while True:
      ret, frame = cap.read()

      if mode == "ss":
        extractor = SsObjExtractor(img=frame, padding=padding, is_fast=ss_is_fast, num_rects=num_rects, min_area=min_area, max_area=max_area)
      elif mode == "canny":
        extractor = CannyObjExtractor(img=frame, padding=padding, num_rects=num_rects, sigma=canny_sigma, min_area=min_area, max_area=max_area)
      elif mode == "sfg":
        extractor = SaliencyFineGrainedExtractor(img=frame, padding=padding, sigma=canny_sigma, num_rects=num_rects, min_area=min_area, max_area=max_area)

      extractor.exec()

      callback_params = {
        'extractor': extractor,
        'extractor_util': ExtractorUtil(extractor),
        'image': frame,
        'output_dir': output_dir
      }

      # Save data in tracker
      tracker.snapshot(extractor)

      tracker.print_data()

      cv2.imshow(WINDOW_NAME, extractor.processed_img)
      cv2.setMouseCallback(WINDOW_NAME, mouse_callback, callback_params)
      
      # Record key press
      k = cv2.waitKey(1) & 0xFF

      # Press 'q' to quit
      if(k == KEY_Q):
        break
    
    cap.release()
  else:
    image = cv2.imread(input_file)

    if mode == "ss":
      extractor = SsObjExtractor(img=image, padding=padding, is_fast=ss_is_fast, num_rects=num_rects, min_area=min_area, max_area=max_area)
    elif mode == "canny":
      extractor = CannyObjExtractor(img=image, padding=padding, num_rects=num_rects, sigma=canny_sigma, min_area=min_area, max_area=max_area)
    elif mode == "sfg":
      extractor = SaliencyFineGrainedExtractor(img=image, padding=padding, sigma=canny_sigma, num_rects=num_rects, min_area=min_area, max_area=max_area)

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
