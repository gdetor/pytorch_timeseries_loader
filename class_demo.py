# An example of how to use the Pytorch TimeseriesLoader class for iterating
# over time series data sets (univariate and multivariate).
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

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pylab as plt

from timeseries_loader import TimeseriesLoader
# from transforms import TimeWarp, MagnitudeWarp, Scale, Jitter


if __name__ == '__main__':
    seq_len = 12
    horizon = 1
    data_path = None
    batch_size = 1

    t = np.linspace(0, 1, 1000)
    Y = np.sin(2.*np.pi*t*15) + np.random.normal(0, 0.1, len(t))

    ts = TimeseriesLoader(data_path,
                          data=Y,
                          whole_seq=True,
                          power_transform=True,
                          sequence_len=seq_len,
                          horizon=horizon,
                          # transforms=[Jitter()],
                          transforms=None,
                          n_features=1)

    ts_data = DataLoader(ts, batch_size=batch_size, shuffle=False,
                         drop_last=True)

    gen = iter(ts_data)
    for i in range(10):
        x, y, idx = next(gen)
        print(x.shape, y.shape, idx)

    xx = np.arange(12)
    plt.plot(xx, Y[9:9+seq_len], 'green', ms=10, label="raw signal")
    plt.plot(xx, x[0, :, 0], '-kx', ms=10, label='input x')
    xx = np.arange(1, 13)
    plt.plot(xx, y[0, :, 0], '-ro', label='target y')
    plt.legend()
    plt.show()
