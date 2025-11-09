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

    feature_grupes = sorted(list(set(col.split('_')[0] for col in features.columns)))
    print(feature_grupes)

    num_of_tests = 10
    acc = []
    for prefix in feature_grupes:
        
        selected_columns = [col for col in features.columns if col.startswith(prefix)]
        selected_features = features[selected_columns].to_numpy()
    
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

        print(f"{prefix}, nf:{len(selected_columns)}, acc:{acc[-1]}")


    best_features_idx = np.argmax(np.array(acc))
    print(f"best accuracy: {acc[best_features_idx]}; features: {feature_grupes[best_features_idx]}")



# CM - Co-occurrence Matrix
# DZM - Distance Zone Matrix
# IH - Intensity Histogram
# IVH - Intensity Variance Histogram
# LOC - Local intensity features
# Morph - Morphological Features
# NGL - Neighbourhood Gray-Level Dependence Matrix
# NGT - Neighbourhood Gray-Level Texture
# RLM - Run Length Matrix
# Stat - Statistical Features
# SZM - Size Zone Matrix



#        Old-------------------------- New----------------------------
# class  nf     acc                    nf      acc
# cm,    nf:25, acc:0.5758620689655173 nf:26,  acc:0.5804597701149425
# dzm,   nf:16, acc:0.6839080459770115 nf:16,  acc:0.48965517241379314
# ih,    nf:23, acc:0.5804597701149425 nf:23,  acc:0.5333333333333333
# ivh,   nf:14, acc:0.564367816091954  nf:15,  acc:0.5620689655172414
# loc,   nf:2,  acc:0.5367816091954023 nf:2,   acc:0.5931034482758621
# morph, nf:23, acc:0.5517241379310345 nf:27,  acc:0.6252873563218391
# ngl,   nf:17, acc:0.603448275862069  nf:17,  acc:0.5977011494252873
# ngt,   nf:5,  acc:0.632183908045977  nf:5,   acc:0.6597701149425287
# rlm,   nf:16, acc:0.5206896551724138 nf:16,  acc:0.6310344827586206
# stat,  nf:17, acc:0.5367816091954023 nf:18,  acc:0.5011494252873564
# szm,   nf:16, acc:0.6781609195402301 nf:16,  acc:0.6873563218390805


# filter nf(across all features classes) acc
# lbp,   nf:154,                         acc:0.4896551724137931


