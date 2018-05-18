import matplotlib
from matplotlib import pyplot
import pandas
import tensorflow as tf
from tensorflow.python.data import Dataset
import numpy as np

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

newsDataFrame = pandas.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
print(newsDataFrame.keys())
pyplot.scatter(newsDataFrame[' is_weekend'], newsDataFrame[' shares'])
pyplot.savefig("../../../public_html/plot.png")

feature_columns = [tf.feature_column.numeric_column(' is_weekend')]

my_feature = newsDataFrame[[" is_weekend"]]

targets = newsDataFrame[" shares"]

linreg = tf.estimator.LinearRegressor(feature_columns)

linreg.train((lambda : my_input_fn(my_feature, targets, 10)), steps = 1000)

# pyplot.scatter(pyplot.predict(lambda : my_input_fn(, newsDataFrame[' shares'], 100)))

pyplot.savefig("../../../public_html/predictions.png")
