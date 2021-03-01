import os
import numpy as np
from sklearn.metrics import confusion_matrix
from math import sqrt
import cv2
from torch import tensor
import torch
import torch.nn as nn

def load_images_from_dir(dir, img_width, img_height):
  images = []

  for filename in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, filename))
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255

    if img is not None:
      images.append(img)

  return images

def cv2_to_tensor(images):
  x = []

  for img in images:
    img_tensor = torch.tensor(img.transpose((2,0,1))).float()
    d = img_tensor.detach().numpy()
    x.append(d)

  return torch.tensor(x).float()

def performance_metrics(validation_labels, predictions):
  tn, fp, fn, tp = np.array(confusion_matrix(validation_labels, predictions).ravel(), dtype=np.float64)

  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  ppv = tp / (tp + fp)
  npv = tn / (tn + fn)
  ts  = tp / (tp + fn + fp)
  pt  = (sqrt(tpr * (-tnr + 1)) + tnr - 1) / (tpr + tnr - 1)
  f1  = tp / (tp + 0.5 * (fp + fn))
  acc = (tp + tn) / (tp + tn + fp + fn)

  mcc_denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  if mcc_denominator > 0:
    mcc = ((tp * tn) - (fp * fn))  / mcc_denominator
  else:   
    mcc = -1

  return tp, tn, fp, fn, tpr, tnr, ppv, npv, ts, pt, acc, f1, mcc

def htb(data):
  outp = []

  def htb_inner(data):
    """
    Inner ht breaks function for recursively computing the break points.
    """
    data_length = float(len(data))
    data_mean   = sum(data) / data_length
    head = [_ for _ in data if _ > data_mean]
    outp.append(data_mean)

    while len(head) > 1 and len(head) / data_length < 0.40:
      return htb_inner(head)

  htb_inner(data)

  return outp

def fetch_threshold(bins, counts, break_point):
  index       = 0
  latest_min  = 99999999
  threshold   = -1

  for i in range(len(counts)):
    if abs(counts[i] - break_point) <= latest_min:
      latest_min = abs(counts[i] - break_point)
      index = i
      threshold = ((bins[i + 1] - bins[i]) / 2) + bins[i]
      #threshold = bins[i + 1]
      #threshold = bins[i]

  return threshold

def create_histogram(data, num_bins=100, step=-1):
  min_bin = np.min(data)
  max_bin = np.max(data) + min_bin

  if step < 0:
    step  = (max_bin - min_bin) / num_bins

  bins  = np.arange(min_bin, max_bin, step)

  (hist, bins) = np.histogram(data, bins=bins)

  return (hist,bins)
