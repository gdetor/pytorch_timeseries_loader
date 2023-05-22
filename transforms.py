# Data augmentation transforms for time series.
# Copyright (C) 2023 Georgios Is. Detorakis (gdetor@protonmail.com)
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

import numpy as np
from scipy.interpolate import CubicSpline

# np.random.seed(123)


class Jitter:
    """
    Time series augmentation by jittering
    """

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        epsilon = np.random.normal(loc=0.0, scale=self.sigma, size=x.shape)

        x += epsilon
        y += epsilon

        return x, y


class Scale:
    """
    Time series augmentation by amplitude scaling.
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        scale_factor = np.random.normal(0, self.sigma, (x.shape))

        x *= scale_factor
        y *= scale_factor

        return x, y


class Permutation:
    """
    Time series augmentation by permuting

    Temporal dependencies are not conserved
    """

    def __init__(self, sequence_len=1, num_intervals=3):
        if num_intervals > 5:
            num_intervals = 5
        if num_intervals <= 1:
            print("meaningless permutation transform")
            exit()
        self.interval_len = int(sequence_len // num_intervals)
        self.index = np.arange(sequence_len).reshape(-1, self.interval_len)

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        np.random.shuffle(self.index)

        x = x[self.index.flatten()]
        y = y[self.index.flatten()]

        return x, y


class Rotation:
    """
    Time series augmentation by rotating (flipping) a univariate time series.
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        if x.shape[-1] == 1:
            x = x[::-1, :]
            y = y[::-1, :]

        return x, y


class MagnitudeWarp:
    """
    Time series augmentation by using the method of magnitude warping.
    """
    def __init__(self, sigma=0.1, num_knots=12):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        sequence_len, n_features = x.shape

        knots = np.random.normal(1.0,
                                 self.sigma,
                                 (self.num_knots+2, n_features))
        steps = (np.ones((n_features, 1)) * (np.linspace(0,
                                                         sequence_len-1.,
                                                         self.num_knots+2))).T
        X = np.arange(sequence_len)
        warper = np.array([CubicSpline(steps[:, dim],
                                       knots[:, dim])(X)
                           for dim in range(n_features)]).T

        # x_ = x * warper
        # y_ = y * warper
        x *= warper
        y *= warper
        return x, y


class TimeWarp:
    """
    Time series augmentation by using the method of time warping.
    """
    def __init__(self, sigma=0.2, num_knots=8):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        sequence_len, n_features = x.shape[0], x.shape[1]

        X = np.arange(sequence_len)
        random_warps = np.random.normal(1.0,
                                        self.sigma,
                                        (self.num_knots+2, n_features))
        steps = (np.ones((n_features, 1)) * (np.linspace(0,
                                                         sequence_len-1.,
                                                         self.num_knots+2))).T
        for dim in range(n_features):
            time_warp = CubicSpline(steps[:, dim],
                                    steps[:, dim]
                                    * random_warps[:, dim])(X)
            scale = (sequence_len - 1) / time_warp[-1]
            x[:, dim] = np.interp(X, np.clip(scale * time_warp,
                                  0,
                                  sequence_len - 1), x[:, dim]).T
            y[:, dim] = np.interp(X, np.clip(scale * time_warp,
                                  0,
                                  sequence_len - 1), y[:, dim]).T
        return x, y
