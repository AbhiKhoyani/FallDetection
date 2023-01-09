import imblearn
import wandb
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame
from collections import Counter

def generate_signal(array):
  assert array.ndim == 1
  mu, sigma = np.mean(array), np.std(array)
  return np.random.normal(0, sigma, size = array.shape)

def mu_sigma_DA(X, y):
  np.random.seed(1234)
  # assert type(X)== 'numpy.ndarray', 'X type should be numpy array' 
  # assert X.ndim == 3, 'X should be 3 Dimensional'
  total_samples, time = X.shape
  final_plus = []
  final_minus = []
  for n in range(total_samples):
    temp = X[n,:]
    noise = generate_signal(temp)
    sample_plus = temp+noise
    sample_minus = temp-noise
    
    final_plus.append(np.array(sample_plus).T)
    final_minus.append(np.array(sample_minus).T)
  # final = np.vstack(final)
  return np.vstack([X, np.array(final_plus), np.array(final_minus)]), np.hstack([y, y, y])

def SMOTE_oversampling(X, y):
    
    sampler = imblearn.over_sampling.SMOTE()
    X, y = sampler.fit_resample(X, y)
    print('Classe Number after Oversampling is: ',Counter(y))
    return X, y

def load_dataset(ms, smote):
  artifact = wandb.use_artifact('abhi_khoyani/Fall-Detection/split_data:v0', type='split_data')
  table = artifact.get('split_data')

  # split into train-test-val split as per "Stage" column 
  df = DataFrame(data = table.data, columns=table.columns)
  X_train, y_train = np.vstack(df.loc[df.Stage == 'train'].X), df.loc[df.Stage == 'train'].label.to_numpy()
  X_val, y_val = np.vstack(df.loc[df.Stage == 'valid'].X), df.loc[df.Stage == 'valid'].label.to_numpy()
  X_test, y_test = np.vstack(df.loc[df.Stage == 'test'].X), df.loc[df.Stage == 'test'].label.to_numpy()
  if smote:
    X_train, y_train = SMOTE_oversampling(X_train, y_train)

  if ms:
    X_train, y_train = mu_sigma_DA(X_train, y_train)
  
  y_train = np.expand_dims(y_train, -1)
  y_val = np.expand_dims(y_val, -1)
  y_test = np.expand_dims(y_test, -1)

  return X_train, y_train, X_val, y_val, X_test, y_test