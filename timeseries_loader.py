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


from numpy import load, float32, random, expand_dims, array
from numpy import nan_to_num, log, abs, sign, isnan, count_nonzero
from scipy.stats import boxcox

from torch import from_numpy, is_tensor
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def muLaw(x, mu=255):
    """! Performs the mu-law encoding for a given input x and a parameter mu.
    Essentially it reduces the dynamic range of the input signal x.

    @param x Input vector (ndarray)
    @param mu The mu compression parameter (int)
    """
    tmp = sign(x) * log(1 + mu * abs(x)) / log(1 + mu)
    return tmp


def invMuLaw(y, mu=255):
    """! Performs the inverse mu-law encoding for a given input y and a
    parameter mu.

    @param y Input vector with values in [-1, 1] (ndarray)
    @param mu The mu compression parameter (int)
    """
    tmp = sign(y) * (((1 + mu)**abs(y) - 1.0) / mu)
    return tmp


class TimeseriesLoader(Dataset):
    """! A Pytorch DataLoader iterator for time series data sets. This class
    provides, as well, the basic transformations for time series data sets.
    """
    def __init__(self,
                 data_path=None,
                 data=None,
                 train=True,
                 data_split_perc=0.7,
                 entire_seq=False,
                 sequence_length=10,
                 horizon=1,
                 dim_size=1,
                 scale=False,
                 scale_intvl=[0, 1],
                 standarize=False,
                 power_transform=False,
                 noise=False,
                 var=1.0,
                 mulaw=False,
                 mu=255):
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
        @param entire_seq When enabled the iterator will return the entire
        target sequence and not only one single target point (bool)
        @param sequence_length The length of the sequence used as input (int)
        @param horizon How many points in the future the iterator will return
        as target (int)
        @param dim_size The dimension of the features (univariate time series
        have dim=1, mutlivariate dim=n) (int)
        @param scale Normalizes the data into interval scale_intvl (bool)
        @param scale_intvl Normalization interval (list)
        @param standarize   Standarizes the data (bool)
        @param power_transform  Transforms the data using a Box-Cox transform
        (bool)
        @param noise Adds white noise with zero mean to the data (bool)
        @param var  The variance of the added white noise
        @param mulaw Performs a mu-law transformation (bool)
        @param mu The parameter mu of the mu-law

        @note The user can either specify the path where the data are stored
        as numpy files or they can directly provide a numpy array with the
        raw data.

        @return void
        """

        self.ent_sequence = entire_seq
        self.horizon = horizon
        self.split_perc = data_split_perc
        self.mu = mu

        self.scaler = None
        self.standarized = None
        self.boxcox = None

        # Load the data
        if data_path is None:
            data = array(data).astype(float32)
        else:
            data = load(data_path).astype(float32)
        if data.ndim != 1:
            data = data[:, :dim_size]

        # Check for NaNs
        if count_nonzero(isnan(data)):
            print("WARNING: NaN detected in the raw data!")

        # Remove NaNs
        data = nan_to_num(data, nan=0.0).astype(float32)

        # Split the data (train / test)
        ntrain_data = int(data.shape[0] * self.split_perc)
        if train is True:
            data = data[:ntrain_data]
        else:
            data = data[ntrain_data:]
        self.shape = data.shape             # Keep data shape
        self.win_len = sequence_length      # Prediction horizon

        # Add white noise to the data
        if noise is True:
            data += random.normal(0, var, data.shape)

        # Convert data Numpy array to Torch Tensor
        self.data = from_numpy(data)

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
            self.data = self.data.astype(float32)

        # Standarize the data (x - mu) /  sigma
        if standarize is True:
            self.standarizer = StandardScaler()
            if len(self.data.shape) == 1:
                self.data =\
                       self.standarizer.fit_transform(self.data.reshape(-1, 1))
                self.data = self.data[:, 0]
            else:
                self.data = self.standarizer.fit_transform(self.data)
            self.data = self.data.astype(float32)

        # Apply a Box-Cox (power) transform
        if power_transform is True:
            self.data, lamda = boxcox(self.data.flatten())
            self.data = self.data.reshape(self.shape)
            self.boxcox = lamda

        # Apply a mu-law algorithm
        if mulaw is True:
            self.data = muLaw(self.data, self.mu)

        # Final data tensor length
        self.size = len(self.data) - (sequence_length + 1)

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

    def get_boxcox(self):
        """! Returns the lambda (that maximizes the log-likelihood function) of
        the Box-Cox power transformation.

        @return self.lamda The value of the lambda parameter of the Box-Cox
        transform (float)
        """
        return self.boxcox

    def get_mu(self):
        """! Returns the mu of the mu-law transformation.

        @return self.mu The used mu value of the mu-law algorithm (int)
        """
        return self.mu

    def __len__(self):
        """! Returns the length of data Tensor.

        @return len(self.data) The temporal length of the raw data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """! Gets an item from Tensor data. Slide the window based on the
        horizon.

        @param idx The time index of the data provided by the DataLoader class
        of Pytorch upon request during training/testing.

        @return A tuple (x, y, idx) where x is the input data, y is the target
        and idx the time index
        """
        if is_tensor(idx):
            idx = idx.tolist()
        idx %= (self.size - self.horizon)
        # idx %= self.size
        # It returns the entire sequence from x_[idx + 1] to x_[idx+sequence+1]
        if self.ent_sequence is False:
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
            x = expand_dims(x, axis=1)
            y = expand_dims(y, axis=1)
        return x, y, idx
