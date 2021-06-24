import tensorflow_datasets as tfds


dataset, info = tfds.load('oxford_iiit_pet:3.*.*', data_dir="../../data", with_info=True)

for item in dataset["train"].take(1):
    print(item)