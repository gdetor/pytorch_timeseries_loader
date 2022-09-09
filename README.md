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
                      sequence_length=seq_len,
                      horizon=horizon,
                      dim_size=1)
                      
# Pass the ts instance to the Pytorch's DataLoader class
ts_data = DataLoader(ts, batch_size=batch_size, shuffle=False,
                     drop_last=True)
```

For more details about all the available parameters and methods,
please see *timeseries_loader.py*. 

## Dependencies
  - Numpy
  - Scipy
  - Torch
  - Sklearn
