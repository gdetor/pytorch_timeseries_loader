# A Pytorch class for iterating over time series data sets (univariate and
# multivariate).
# Copyright (C) 2022 Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple

import numpy as np
from scipy.stats import boxcox

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def muLaw(x, mu=255):
    """! Performs the mu-law encoding for a given input x and a parameter mu.
    Essentially it reduces the dynamic range of the input signal x.

    @param x Input vector (ndarray)
    @param mu The mu compression parameter (int)
    """
    tmp = np.sign(x) * np.log(1 + mu * abs(x)) / np.log(1 + mu)
    return tmp


def invMuLaw(y, mu=255):
    """! Performs the inverse mu-law encoding for a given input y and a
    parameter mu.

    @param y Input vector with values in [-1, 1] (ndarray)
    @param mu The mu compression parameter (int)
    """
    tmp = np.sign(y) * (((1 + mu)**abs(y) - 1.0) / mu)
    return tmp


class TimeseriesLoader(Dataset):
    """! A Pytorch DataLoader iterator for time series data sets. This class
    provides, as well, the basic transformations for time series data sets.
    """
    def __init__(self,
                 data_path: str = None,
                 data: np.ndarray = None,
                 train: bool = True,
                 data_split_perc: float = 0.7,
                 whole_seq: bool = False,
                 sequence_len: int = 10,
                 horizon: int = 1,
                 n_features: int = 1,
                 scale: bool = False,
                 scale_intvl: Tuple[float] = [0, 1],
                 standarize: bool = False,
                 power_transform: bool = False,
                 differencing: bool = False,
                 noise: bool = False,
                 var: float = 1.0,
                 mulaw: bool = False,
                 mu: int = 255,
                 transforms=None,
                 transform_prob: float = 0.5):
        """! Constructor method of TimeseriesLoader class. It performs several
        tasks to pre-process the data such as NaNs detection and removal,
        normalization, standarization, power-transformation, and mu-law
        transformation.

        The data must be given as a Numpy array of shape
        (n_samples, n_features)

        @param data_path The path where the data are stored as npy file (str)
        @param data A numpy array that contains the raw data (ndarray)
        @param train It determines if the training data will be used (bool)
        @param data_split_perc  How many data points are used for
        training/testing (float)
        @param whole_seq When enabled the iterator will return the whole
        target sequence and not only one single target point (bool)
        @param sequence_len The length of the sequence used as input. How many
        data points from the past (historical data) will be used for
        training/testing (int)
        @param horizon How many points in the future the iterator will return
        as target sequence - forecasting horizon (int)
        @param n_features The number of features (univariate time series
        have n_features=1, mutlivariates n_features=n) (int)
        @param scale Normalizes the data into interval scale_intvl (bool)
        @param scale_intvl Normalization interval (list)
        @param standarize   Standarizes the data (bool)
        @param power_transform  Transforms the data using a Box-Cox transform
        (bool)
        @param differencing Computes the derivative (difference) of the raw
        data (i.e., stationarity transformation)
        @param noise Adds white noise with zero mean to the data (bool)
        @param var  The variance of the added white noise
        @param mulaw Performs a mu-law transformation (bool)
        @param mu The parameter mu of the mu-law
        @param transforms A list of data augmentation transforms (e.g.,
        [Jitter(), Scale(sigma=0.2). By default is None and no data
        augmentation is enabled.
        @param transform_prob A probability with which a data augmentation
        transform is applied to the raw data.

        @note The user can either specify the path where the data are stored
        as numpy files or they can directly provide a numpy array with the
        raw data.

        @return void
        """

        self.whole_seq = whole_seq
        self.horizon = horizon
        self.split_perc = data_split_perc
        self.mu = mu

        self.scaler = None
        self.standarized = None
        self.boxcox = None

        if transforms is not None:
            self.augmentation = True
            self.transforms = transforms
            self.transform_prob = transform_prob
            if isinstance(self.transforms, list) is False:
                print("Transformation not found")
                exit()
        else:
            self.augmentation = False

        # Load the data
        if data_path is None:
            self.data = np.array(data).astype(np.float32)
        else:
            self.data = np.load(data_path).astype(np.float32)
        if self.data.ndim != 1:
            self.data = self.data[:, :n_features]

        # Check for NaNs
        if np.count_nonzero(np.isnan(data)):
            print("WARNING: NaN detected in the raw data!")

        # Remove NaNs
        self.data = np.nan_to_num(data, nan=0.0).astype(np.float32)

        # Split the data (train / test)
        self.ntrain_data = int(data.shape[0] * self.split_perc)
        if train is True:
            self.data = self.data[:self.ntrain_data]
        else:
            self.data = self.data[self.ntrain_data:]
        self.shape = self.data.shape        # Keep data shape
        self.win_len = sequence_len         # Sequence length (historical data)

        # Add white noise to the data
        if noise is True:
            self.data += np.random.normal(0, var, data.shape)

        # Ensure the data are positive when Box-Cox transform is enabled
        if scale is False and power_transform is True:
            scale = True
            if standarize is True:
                standarize = False
            if scale_intvl[0] <= 0:
                scale_intvl[0] += 0.001

        # Scale the data [0, 1]
        if scale:
            self.scaler = MinMaxScaler(feature_range=(scale_intvl[0],
                                                      scale_intvl[1]),
                                       copy=True)
            if len(self.data.shape) == 1:
                self.data = self.scaler.fit_transform(self.data.reshape(-1, 1))
                self.data = self.data[:, 0]
            else:
                self.data = self.scaler.fit_transform(self.data)
            self.data = self.data.astype(np.float32)

        # Standarize the data (x - mu) /  sigma
        if standarize is True:
            self.standarizer = StandardScaler()
            if len(self.data.shape) == 1:
                self.data =\
                       self.standarizer.fit_transform(self.data.reshape(-1, 1))
                self.data = self.data[:, 0]
            else:
                self.data = self.standarizer.fit_transform(self.data)
            self.data = self.data.astype(np.float32)

        # Apply a Box-Cox (power) transform
        if power_transform is True:
            self.data, lamda = boxcox(self.data.flatten())
            self.data = self.data.reshape(self.shape)
            self.boxcox = lamda

        # Apply a mu-law algorithm
        if mulaw is True:
            self.data = muLaw(self.data, self.mu)

        # Apply a differencing transform
        if differencing is True:
            self.data = np.diff(self.data, axis=0)

        # Convert data Numpy array to Torch Tensor
        self.data = torch.from_numpy(data)

        # Final data tensor length
        self.size = len(self.data) - (sequence_len + 1)

    def apply_transforms(self,
                         x: torch.tensor,
                         y: torch.tensor) -> Tuple[torch.tensor]:
        """! Applies a data augmentation transform on the raw data. The
        augmentation occurs when a random number drawn from a uniform
        distribution in [0, 1) is smaller than a predefined probability
        (transform_prob). Otherwise, the data remain unchanged.

        @return The augmented (transformed) inputs (x) and the targets (y).
        """
        for trans in self.transforms:
            if np.random.uniform(0, 1) < self.transform_prob:
                x, y = trans([x, y])
        return x, y

    def get_scaler(self):
        """! Returns the MInMaxScaler object of scikit-learn in case the user
        needs to inverse the normalization later.

        @return self.scaler Normalization scaler object
        """
        return self.scaler

    def get_standarized(self):
        """! Returns the StandardScaler object of scikit-learn in case the user
        needs to inverse the standarization later.

        @return self.standarizer Standarization scaler object
        """
        return self.standarizer

    def get_boxcox(self) -> float:
        """! Returns the lambda (that maximizes the log-likelihood function) of
        the Box-Cox power transformation.

        @return self.lamda The value of the lambda parameter of the Box-Cox
        transform (float)
        """
        return self.boxcox

    def get_mu(self) -> int:
        """! Returns the mu of the mu-law transformation.

        @return self.mu The used mu value of the mu-law algorithm (int)
        """
        return self.mu

    def __len__(self) -> int:
        """! Returns the length of data Tensor.

        @return len(self.data) The temporal length of the raw data
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, int]:
        """! Gets an item from Tensor data. Slide the window based on the
        horizon.

        @param idx The time index of the data provided by the DataLoader class
        of Pytorch upon request during training/testing.

        @return A tuple (x, y, idx) where x is the input data, y is the target
        and idx the time index. The shape of the tensors is
        (batch_size, sequence_len, n_features) or
        (batch_size, horizon, n_features) in case the whole_seq flag is False
        and not the whole sequence is needed.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx %= (self.size - self.horizon)
        # idx %= self.size
        # It returns the whole sequence from x_[idx + 1] to x_[idx+sequence+1]
        if self.whole_seq is False:
            startpnt = idx
            endpnt = idx + self.win_len
            x = self.data[startpnt:endpnt]
            y = self.data[endpnt:(endpnt + self.horizon)]
        else:
            startpnt = idx
            endpnt = idx + self.win_len
            x = self.data[startpnt:endpnt]
            y = self.data[(startpnt + self.horizon):(endpnt + self.horizon)]

        if self.data.ndim == 1:
            x = np.expand_dims(x, axis=1)
            y = np.expand_dims(y, axis=1)

        if self.augmentation:
            x, y = self.apply_transforms(x, y)
        return x.copy(), y.copy(), idx


def split_timeseries_data(data,
                          train_data_perc: float = 0.8,
                          sequence_len: int = 12,
                          horizon: int = 1,
                          is_univariate: bool = False,
                          is_torch: bool = True) -> Tuple[torch.tensor]:
    """! This function serves the simple purpose of splitting the time
    series data into training and testing sets. It accepts the raw data
    in a Numpy array and returns Torch tensors with the training and
    testing data.

    Example 1:
    X_train, y_train, X_test, y_test = split_timeseries_data(data)

    step = int(X_train.shape[0] // batch_size)

    for e in range(epochs):
        for i in range(step):
            x = X_train[i*batch_size:(i+1) * batch_size]
            y = y_train[i*batch_size:(i+1) * batch_size]

    The user can use the tensors (X_train, y_train) and (X_test, y_test) with
    the DataLoader class of Pytorch. To do so, the user must first obtain the
    tensors using the function split_timeseries_data() and then use the
    TensorDataset method of Pytorch. Then, they can pass the new tensors as an
    argument to the DataLoader.

    Example 2:
    X_train, y_train, X_test, y_test = split_timeseries_data(data)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)

    @param train_data_perc represents the percentage of data used for training,
    with the remainder used for testing.
    @param sequence_len The length of the input sequence (past data) (int)
    @param horizon How many data points into the future should the target
    vector contains (int)
    @param univariate If enabled, the dimension of the features is removed from
    the tensors (bool). The default value is False
    @param torch If enabled, the Numpy vectors will be converted to torch
    tensors (bool). The default value is True

    @note The user is responsible for normalizing, standardizing, or
    transforming the raw data. This function does not perform any such
    transformation on the data.

    @return A Python tuple with the X_train, y_train (training input/targets)
    and X_test, y_test (testing input / targets). The tensors are of
    shape (n_samples, sequence_len, num_features). When univariate is True the
    tensors are of shape (batch_size, sequence_len).
    """
    if data.ndim == 1:
        m = data.shape[0]
        n = 1
    else:
        m, n = data.shape
    n_train = int(m * train_data_perc)
    n_test = m - n_train

    data_train = data[:n_train].astype('float32')
    data_test = data[n_train:].astype('float32')

    train_size = n_train - sequence_len - horizon
    if train_size <= sequence_len:
        raise ValueError("Sequence length is too large!")
    X_train = np.zeros((train_size, sequence_len, n), 'float32')
    y_train = np.zeros((train_size, sequence_len, n), 'float32')
    if n == 1:
        for i in range(n_train-sequence_len-horizon):
            X_train[i, :, 0] = data_train[i:i + sequence_len]
            y_train[i, :, 0] = data_train[i+horizon:i+sequence_len+horizon]
            # y_train[i, :, 0] = data_train[seq_len+i:seq_len+i+horizon]
    else:
        for i in range(n_train-sequence_len-horizon):
            X_train[i] = data_train[i:i + sequence_len]
            y_train[i] = data_train[i + horizon:i + sequence_len + horizon]

    if is_univariate:
        X_train = X_train[:, :, 0]
        y_train = y_train[:, -horizon:, 0]

    if is_torch:
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

    test_size = n_test - sequence_len - horizon
    if test_size <= sequence_len:
        raise ValueError("Sequence length is too large!")
    X_test = np.zeros((test_size, sequence_len, n), 'float32')
    y_test = np.zeros((test_size, sequence_len, n), 'float32')
    if n == 1:
        for i in range(n_test-sequence_len-horizon):
            X_test[i, :, 0] = data_test[i:i + sequence_len]
            y_test[i, :, 0] = data_test[i + horizon:i + sequence_len + horizon]
    else:
        for i in range(n_test-sequence_len-horizon):
            X_test[i] = data_test[i:sequence_len + i]
            y_test[i] = data_test[i + horizon:i + sequence_len + horizon]

    if is_univariate:
        X_test = X_test[:, :, 0]
        y_test = y_test[:, -horizon:, 0]

    if torch:
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
    return X_train, y_train, X_test, y_test
