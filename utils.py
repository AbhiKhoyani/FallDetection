import os
import random
import numpy as np
import tensorflow as tf


def reproducibility(seed = 1234):
  #fix random seed for reproducibility
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  tf.compat.v1.set_random_seed(seed)
  tf.random.set_seed(seed)
  tf.keras.utils.set_random_seed(seed)  # sets seeds for base-python, numpy and tf
  tf.config.experimental.enable_op_determinism()
  
if __name__ == "__main__":
    reproducibility()