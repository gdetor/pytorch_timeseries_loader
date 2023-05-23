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
        return x + epsilon, y + epsilon


class Scale:
    """
    Time series augmentation by amplitude scaling.
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        scale_factor = np.random.normal(loc=1,
                                        scale=self.sigma,
                                        size=(x.shape[1]),)
        return x * scale_factor, y * scale_factor


class Permutation:
    """
    Time series augmentation by permuting

    Temporal dependencies are not conserved
    """
    def __init__(self, num_segments=3):
        if num_segments > 5:
            num_segments = 5
        if num_segments <= 1:
            print("meaningless permutation transform")
            exit()
        self.num_segments = num_segments

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        sequence_len, n_features = x.shape[0], x.shape[1]
        index = np.arange(sequence_len)

        x_perm, y_perm = np.zeros_like(x), np.zeros_like(y)
        for i in range(n_features):
            segments = np.array(np.array_split(index, self.num_segments),
                                dtype=object)
            segments = np.random.permutation(segments)
            permuted_index = np.concatenate(segments).astype('i')
            x_perm[:, i] = x[permuted_index, i]
            y_perm[:, i] = y[permuted_index, i]
        return x_perm, y_perm


class Rotation:
    """
    Time series augmentation by rotating (flipping) a univariate time series.
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, sample):
        x, y = sample
        assert x.shape == y.shape

        flip = np.random.choice([-1, 1], size=(x.shape[1],)).reshape(1, -1)
        rotate_axis = np.arange(x.shape[1])
        np.random.shuffle(rotate_axis)

        return flip * x[:, rotate_axis], flip * y[:, rotate_axis]


class MagnitudeWarp:
    """
    Time series augmentation by using the method of magnitude warping.
    """
    def __init__(self, sigma=0.1, num_knots=5):
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

        return x * warper, y * warper


class TimeWarp:
    """
    Time series augmentation by using the method of time warping.
    """
    def __init__(self, sigma=0.2, num_knots=5):
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
        x_, y_ = np.zeros_like(x), np.zeros_like(y)
        for dim in range(n_features):
            time_warp = CubicSpline(steps[:, dim],
                                    steps[:, dim]
                                    * random_warps[:, dim])(X)
            scale = (sequence_len - 1) / time_warp[-1]
            x_[:, dim] = np.interp(X, np.clip(scale * time_warp,
                                   0,
                                   sequence_len - 1), x[:, dim]).T
            y_[:, dim] = np.interp(X, np.clip(scale * time_warp,
                                   0,
                                   sequence_len - 1), y[:, dim]).T
        return x_, y_


if __name__ == "__main__":
    import matplotlib.pylab as plt

    y = np.array([0.0, 0.5, 1.0, 1.5, 1.0, 1.2, 0.8, 1.2, 0.7, 0.1, 0.1, 0.1,
                  2.0, 1.5, 1.1, 0.1])
    t = np.arange(len(y))

    x = np.linspace(0, len(y), 1000)
    yp = np.interp(x, t, y)

    plt.figure()
    plt.plot(yp)

    B = yp.reshape(10, 100).T
    plt.figure()
    plt.plot(B[:, 5])

    trans = Rotation()
    yt, _ = trans([B[:, 0].reshape(-1, 1),
                   B[:, 0].reshape(-1, 1)])

    plt.figure()
    plt.plot(B[:, 0], label="original")
    plt.plot(yt[:, 0], label="transformed")
    plt.legend()
    plt.show()
