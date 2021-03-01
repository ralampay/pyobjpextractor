class AverageObjectPerceptionsFeatureExtractor:
  def __init__(self, cnn_autoencoder, objects):
    self.cnn_autoencoder  = cnn_autoencoder
    self.objects          = objects
    self.num_objects      = float(len(objects))

  def execute(self):
    features  = [self.num_objects]

    obj_features = self.cnn_autoencoder.conv(self.objects)

    for feature_vector in obj_features:
      for x in feature_vector:
        features.push(x)

    return features
