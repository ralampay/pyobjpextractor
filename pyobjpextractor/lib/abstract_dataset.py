from torch.utils.data import Dataset, DataLoader

class AbstractDataset(Dataset):
  def __init__(self, X):
    self.x = X
    self.y = X
    self.n_samples = len(X)

  def __getitem__(self, index):
    return self.x[index], self.x[index]

  def __len__(self):
    return self.n_samples
