# pytorch_timeseries_loader

TimeseriesLoader is a Pytorch dataset class that provides all the
necessary means to preprocess time series raw data and iterate over
them during training or testing. 


## Available preprocessing methods
Here is a list of all available preprocessing methods provided by
TimeseriesLoader:

  - **Normalization** Normalizes the raw data into an interval [a, b], the
  default interval is [0, 1].
  - **Standardization** Standardizes the raw data using the z-score.
  - **White noise** The user can add white noise with zero mean
  and variable variance.
  - **Power transform** Provides a Box-Cox power-transformation method.
  - **mu-law** The user can apply a mu-law algorithm to compress
  the raw data  (especially if the time series are sound signals).

TimeseriesLoader provides all the necessary methods for the user
to obtain all the objects returned from the transformations,
so if they would like to inverse the transformations later, they can.

## Example usage

The raw data `Y` is a Numpy array of shape `(n_samples, n_features)`.
In the case of a univariate time series, `n_features = 1`. 

```python

# Instantiate TimeseriesLoader class 
ts = TimeseriesLoader(data_path,
                      data=Y,
                      entire_seq=True,
                      power_transform=True,
                      sequence_len=seq_len,
                      horizon=horizon,
                      dim_size=1)
                      
# Pass the ts instance to the Pytorch's DataLoader class
ts_data = DataLoader(ts, batch_size=batch_size, shuffle=False,
                     drop_last=True)
```
For more details about all the available parameters and methods,
please see *timeseries_loader.py*. 

Along with the Pytorch class **TimeseriesLoader**, we provide a simpler
function called **split_timeseries_data** which takes as input raw time series
data along with the length of the historical (past) data sequence and the
forecasting horizon, and returns a Python tuple of training and testing 
torch tensors. 

```python
    X_train, y_train, X_test, y_test = split_timeseries_data(data)

    step = int(X_train.shape[0] // batch_size)

    for e in range(epochs):
        for i in range(step):
            x = X_train[i*batch_size:(i+1) * batch_size]
            y = y_train[i*batch_size:(i+1) * batch_size]
```

Another way the user can utilize the function `split_timeseries_data` is to
use the tensors `(X_train, y_train)` and `(X_test, y_test)` with the 
`DataLoader` class of Pytorch. To do so, the user must first obtain the tensors
using the function `split_timeseries_data()` and then use the `TensorDataset`
method of Pytorch. Then, they can pass the new tensors as an argument to the
`DataLoader`.

```python
X_train, y_train, X_test, y_test = split_timeseries_data(data)

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
```


Furthermore, the user is responsible for normalizing, standardizing, or 
transforming their raw data. The function *split_timeseries_data* does not
perform any such transformation on the data. It returns a Python tuple with
the `X_train`, `y_train` (training input/targets) and `X_test`, `y_test`
(testing input / targets). The tensors are of shape 
`(n_samples, sequence_len, num_features)`.


## Data Augmentation - Transforms

Along with the TimeSeriesLoader class, we provide some fundamental data
augmentation transformations for time series, such as jitter, scale,
permutation, flipping, time warping, and magnitude warping.
The enterprising user can find many more transforms implemented in Python
[here]https://github.com/uchidalab/time_series_augmentation)
and can read more in [1] and [2]. 
All the transforms provided here are implemented in the file **transforms.py**
as Python class that can be used with Python (or Pytorch). 

The user can pass any transform (or many) as an argument (list) to the
**TimeSeriesLoader** class. For instance, 
```python

ts = TimeseriesLoader(data_path,
                      data=Y,
                      entire_seq=True,
                      power_transform=True,
                      sequence_len=seq_len,
                      horizon=horizon,
                      transforms=[Jitter(sigma=0.5),
                      			  Permutation()],  # Pass the transforms as a list or None if you don't want to apply any data augmentation
                      transform_prob=0.5,
                      dim_size=1)
```

Internally, the class will draw a random number from a uniform distribution
in [0, 1) and if that number is smaller than a given probability (transform_prob)
will apply the transform(s). 




## Dependencies
  - Numpy
  - Scipy
  - Torch
  - Sklearn


## References

  1. B. K. Iwana, and S. Uchida. *An empirical survey of data augmentation
  for time series classification with neural networks.* Plos one 16.7 (2021):
      e0254841.
  2. T. T. Um, et al. "Data augmentation of wearable sensor data for
  parkinson’s disease monitoring using convolutional neural networks."
  Proceedings of the 19th ACM international conference on multimodal
  interaction. 2017.
