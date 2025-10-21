import numpy as np

from config.config import DATASETS
from feature_loader import load_features, split_numpy_data


def distance(x, y, metric='euclidian', p=2):
  if metric == 'euclidian':
    return np.sqrt(np.sum((x[np.newaxis, :] - y) ** 2, axis=-1))
  elif metric == 'max':
    return np.max(np.abs(x[np.newaxis, :] - y), axis=-1)
  elif metric == 'manhattan':
    return np.sum(np.abs(x[np.newaxis, :] - y), axis=-1)
  elif metric == 'p':
    return np.power(np.sum((np.abs(x[np.newaxis, :] - y))**p, axis=-1), 1 / p)

def k_min(x, k):
  return np.argpartition(x,kth=k)[:k]

def knn(x, k, x_train, y_train, metric='euclidian'):
  d = distance(x, x_train, metric)
  nearest = k_min(d, k)
  nearest_labels = y_train[nearest]
  values, counts = np.unique(nearest_labels, return_counts=True)
  return values[np.argmax(counts)]




if __name__ =="__main__":
  classes = [DATASETS.YPD_SD_Acetate_DAY_3_Acetate, DATASETS.YPD_SD_Acetate_DAY_1_Acetate] 


  features, labels = load_features(classes)
  labels = labels.to_numpy()

  # selected_feature_types = ['morph_moran_i', 'morph_geary_c']
  from itertools import combinations
  column_pairs = list(combinations(features.columns, 2))

  acc = []
  for i, selected_feature_types in enumerate(column_pairs):
      if i % 100 == 0:
          print(f"{i}/{len(column_pairs)}")

      selected_features = features.loc[:, selected_feature_types].to_numpy()
      

      # selected_features = selected_features.to_numpy()

      
      num_of_tests = 10
      acc.append(0)
      for test_num in range(num_of_tests):
          x_train, y_train, x_val, y_val, x_test, y_test = split_numpy_data(selected_features, labels, val_percent=0.3, test_percent=0)

          k = 3
          correct = 0
          for input, target in zip(x_val, y_val):
              prediction = knn(input, k, x_train, y_train)
              if prediction == target:
                  correct += 1
          acc[-1] += correct / len(x_val)
      acc[-1] /= num_of_tests

      # print(f"features: {selected_feature_types}; accuracy: {acc[-1]}")

  best_features_idx = np.argmax(np.array(acc))
  print(f"best accuracy: {acc[best_features_idx]}; features: {column_pairs[best_features_idx]}")


  acc = np.array(acc)
  top_9_accs_idxs = np.argpartition(-acc,kth=9)[:9] #-acc, as argpratition returns min

  from matplotlib import pyplot as plt

  fig, axes = plt.subplots(3, 3, figsize=(9, 9))
  for i, idx in enumerate(top_9_accs_idxs):
    x_ax, y_ax = i // 3, i % 3

    x_featyre, y_feature = column_pairs[idx]

    for i_class in classes:
      class_mask = labels == i_class.name
      axes[x_ax, y_ax].scatter(features[class_mask][x_featyre], features[class_mask][y_feature], label=f'{i_class.name}')

    axes[x_ax, y_ax].set_title(f'acc: {acc[idx]}')

    axes[x_ax, y_ax].set_xlabel(x_featyre)
    axes[x_ax, y_ax].set_ylabel(y_feature)

  plt.legend()
  plt.tight_layout()
  plt.show()
       


    