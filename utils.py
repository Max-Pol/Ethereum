import json
import numpy as np


# load the dataset, with the features in <features_loaded>
def load_dataset(features_loaded):
    dataset = []

    with open('local_data/data.json', 'r') as f:
        data = json.load(f)

    for sample in data:
        sample_features = []
        for feature_name in features_loaded:
            sample_features.append(sample[feature_name])
        dataset.append(sample_features)

    return np.array(dataset)
