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
torch tensors. However, the user cannot use this function along with Pytorch's
DataLoader and thus they have to do a little more work in the training/testing
loops. 

```python
    X_train, y_train, X_test, y_test = split_timeseries_data(data)

    step = int(X_train.shape[0] // batch_size)

    for e in range(epochs):
        for i in range(step):
            x = X_train[i*batch_size:(i+1) * batch_size]
            y = y_train[i*batch_size:(i+1) * batch_size]
```

Furthermore, the user is responsible for normalizing, standardizing, or 
transforming their raw data. The function *split_timeseries_data* does not
perform any such transformation on the data. It returns a Python tuple with
the `X_train`, `y_train` (training input/targets) and `X_test`, `y_test`
(testing input / targets). The tensors are of shape 
`(n_samples, sequence_len, num_features)`.

## Dependencies
  - Numpy
  - Scipy
  - Torch
  - Sklearn
