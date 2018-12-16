from __future__ import absolute_import, division, print_function

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#print(dataset_path)

#Read using pandas
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t',
			sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
#print(dataset.isna().sum())

dataset = dataset.dropna()

#Change Origin from categorical to numerical
origin = dataset.pop('Origin')
dataset['USA'] = (origin==1)*1.0
dataset['Europe'] = (origin==2)*1.0
dataset['Japan'] = (origin==3)*1.0
#print(dataset) #dataset.tail() seems to print the tail end entries of the dataset. What if I try to print all?
#printing all will show the beginning and end of the dataset, but will leave the middle out if too long

train_dataset = dataset.sample(frac=0.8, random_state=0) #Not sure what random_state is for
test_dataset= dataset.drop(train_dataset.index) #Very easy way to split dataset into train and test

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde") #Need to learn how to use seaborn
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG") #Pop removes the category or index we give, and we need to remove MPG because thats what the regression needs to predict
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG") #Not sure what these two lines do. Wouldn't the labels and the data be separate entirely?

def norm(x):
	return (x-train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)