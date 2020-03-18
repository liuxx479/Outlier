"""TODO(gaussian_convergence): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np

# TODO(gaussian_convergence): BibTeX citation
_CITATION = """
"""

# TODO(gaussian_convergence):
_DESCRIPTION = """
"""

class GaussianConvergence(tfds.core.GeneratorBasedBuilder):
  """TODO(gaussian_convergence): Short description of my dataset."""

  # TODO(gaussian_convergence): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(gaussian_convergence): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
           "map": tfds.features.Tensor(shape=[256,256], dtype=tf.float32),
           "params": tfds.features.Tensor(shape=[2], dtype=tf.float32)
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("map", "params"),
        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data_path = dl_manager.download_and_extract("https://storage.googleapis.com/ouliers/GRFs.tar")
    label_path = dl_manager.download("https://storage.googleapis.com/ouliers/GRF_params_output.txt")

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
            "images_dir_path": os.path.join(data_path, "home1/02977/jialiu/scratch/Outlier/GRFs"),
            "labels": label_path
            },
        ),
    ]

  def _generate_examples(self, images_dir_path, labels):
    """Yields examples."""

    # First we open the list of cosmological params
    with tf.io.gfile.GFile(labels) as f:
      # Column 1 is om, Column 3 is S8
      table = np.loadtxt(f).astype('float32')

    # Read the maps from the directory
    for i, image_file in enumerate(tf.io.gfile.listdir(images_dir_path)):
      print(i, image_file)
      with tf.io.gfile.GFile(image_file) as f:
        im = fits.getdata(f).astype('float32')
      yield '%d'%i, {"map":im, "params": table[i][[1,3]]}
