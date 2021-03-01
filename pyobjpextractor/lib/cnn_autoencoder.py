import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from torch.utils.data import Dataset, DataLoader
from scipy.stats import iqr
import numpy as np
from scipy import stats

from abstract_dataset import AbstractDataset

from utils import create_histogram
from utils import htb
from utils import fetch_threshold

class CnnAutoencoder(nn.Module):
  def __init__(self, scale=2, channel_maps=[], padding=1, kernel_size=3, num_channels=3, img_width=75, img_height=75, device=torch.device("cpu"), criterion=nn.BCELoss()):
    super().__init__()

    self.device = device

    self.img_width    = img_width
    self.img_height   = img_height
    self.num_channels = num_channels
    self.kernel_size  = kernel_size
    self.padding      = padding
    self.channel_maps = channel_maps
    self.scale        = scale

    self.reversed_channel_maps = list(reversed(channel_maps))

    # Build convolutional layers
    self.convolutional_layers = nn.ModuleList([])

    for i in range(len(self.channel_maps) - 1):
      self.convolutional_layers.append(nn.Conv2d(self.channel_maps[i], self.channel_maps[i+1], kernel_size=self.kernel_size, padding=self.padding))

    # Build deconvolutional layers
    self.deconvolutional_layers = nn.ModuleList([])

    for i in range(len(self.reversed_channel_maps) - 1):
      self.deconvolutional_layers.append(nn.ConvTranspose2d(self.reversed_channel_maps[i], self.reversed_channel_maps[i+1], 2, stride=2))

    self.criterion = criterion

  def conv(self, x):
    for i in range(len(self.convolutional_layers)):
      conv_layer = self.convolutional_layers[i]

      x = F.max_pool2d(F.relu(conv_layer(x)), self.scale)

    return x

  def deconv(self, x):
    for i in range(len(self.deconvolutional_layers)):
      deconv_layer = self.deconvolutional_layers[i]
      x = deconv_layer(x)

      if i != len(self.deconvolutional_layers) - 1:
        x = F.relu(x)
      else:
        x = torch.sigmoid(x)

    return x

  def forward(self, x):
    x = self.conv(x)
    x = self.deconv(x)

    return x

  def errors(self, x):
    x_hat = self.forward(x)

    self.criterion.reduction = 'none'
    err = self.criterion(x_hat.view(-1, self.num_channels * self.img_width * self.img_height), x.view(-1, self.num_channels * self.img_width * self.img_height)).mean(axis=1)
    self.criterion.reduction = 'mean'

    return err.detach().cpu().numpy()

  def predict(self, x):
    errors = self.errors(x)

    bool_arr = errors >= self.optimal_threshold

    return np.array([-1 if elem else 1 for elem in bool_arr])

  def set_optimal_threshold(self, x):
    errors = self.errors(x)

    # Calculate the number of bins according to Freedman-Diaconis rule
    bin_width = 2 * iqr(errors) / np.power(len(errors), (1/3))
    num_bins  = (np.max(errors) - np.min(errors)) / bin_width

    hist, bins = create_histogram(errors, num_bins=num_bins, step=bin_width)
    occurences = [float(o) for o in hist.tolist()]

    breaks = htb(hist)

    possible_thresholds = []

    for b in breaks:
      t = fetch_threshold(bins, hist, b)
      possible_thresholds.append(t)

      self.optimal_threshold = max(possible_thresholds)

    return self.optimal_threshold

  def save(self, filename):
    state = {
      'state_dict': self.state_dict(), 
      'optimizer': self.optimizer.state_dict(), 
      'optimal_threshold': self.optimal_threshold
    }

    torch.save(state, filename)

  def load(self, filename):
    state = torch.load(filename)

    self.load_state_dict(state['state_dict'])

    self.optimizer         = state['optimizer']
    self.optimal_threshold = state['optimal_threshold']

  def fit(self, x, epochs=100, lr=0.001, batch_size=5, with_thresholding=True):
    # Reset errors to empty list
    self.errs = []

    data = AbstractDataset(x)
    dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

    num_iterations = len(x) / batch_size

    for epoch in range(epochs):
      curr_loss = 0

      for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        output = self.forward(inputs)

        loss = self.criterion(output, labels)

        curr_loss += loss
        loss.backward()
        self.optimizer.step()

      curr_loss = curr_loss / num_iterations

      self.errs.append(curr_loss.detach())
      print("=> Epoch: %i\tLoss: %0.5f" % (epoch + 1, curr_loss.item()))

    if with_thresholding:
      print("Setting optimal threshold...")

      self.set_optimal_threshold(x)
