#To generate the Data

import kagglehub

# Download latest version
path = kagglehub.dataset_download("crawford/emnist")

print("Path to dataset files:", path)

# Path to dataset files: /Users/Joshua/.cache/kagglehub/datasets/crawford/emnist/versions/3