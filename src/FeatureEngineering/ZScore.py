"""Module for the ZScore()-class.

Module that stores the ZScore()-class that is
used to compute the ZScore as a feature from
the two cointegrated assets
"""
import numpy as np
import pandas as pd

from src.FeatureEngineering.CointParams import CointParams


class ZScore:
    """Class that handles the generation of the ZScore.

    This class holds multiple methods used to compute
    the Z-Score of a spread of two cointegrated assets.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.gamma = None
        self.mu = None
        self.spread = None
        self.z_score = None
        self.coint_params = CointParams(data)

    def generate_z_score(self,
                         method: str,
                         smoothing: bool,
                         n: int = 120) -> pd.DataFrame:
        """Compute the Z-Score.

        Method that computes the Z-Score which will
        be used as a feature in the NEAT algorithm.
        """
        if method == 'OLS':
            self.mu, self.gamma = self.coint_params.get_params_ols(smoothing=smoothing)
        if method == 'Kalman':
            self.mu, self.gamma = self.coint_params.get_params_kalman(
                smoothing=smoothing
            )
        self.compute_spread(self.gamma, self.mu)
        spread_stats = self.get_ema(self.spread.copy(), n)
        spread_stats['spread_demeaned'] = spread_stats.spread - spread_stats.ema
        spread_stats['spread_demeaned_sq'] = spread_stats.spread_demeaned.apply(
            lambda x: x**2
        )
        spread_stats['spread_var'] = self.get_ema(
            spread_stats.spread_demeaned_sq.to_frame(), n
        )['ema']
        z_score = spread_stats.spread_demeaned / np.sqrt(spread_stats.spread_var)
        self.z_score = z_score
        return z_score

    def compute_spread(self, gamma: pd.DataFrame, mu: pd.DataFrame) -> tuple:
        """Compute the spread of two asset prices.

        This method computes the spread of two assets using the
        cointegration parameters gamma and mu.
        """
        data = self.data.copy()
        w = np.ones((len(data), 2))
        w[:, 0] = w[:, 0] / (1 + gamma)
        w[:, 1] = w[:, 1] * -1 * gamma / (1 + gamma)
        spread = np.sum(np.multiply(data, w), axis=1) - mu / (1 + gamma)
        self.spread = spread.to_frame().rename(columns={0: 'spread'})
        return w, self.spread

    @staticmethod
    def apply_conv_filter(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Apply a one-sided convolutional filter."""
        f = np.array(f)
        x = np.array(x)
        y = np.zeros(len(x))
        y[0 : len(f) - 1] = np.nan
        for i in range(0, len(x)):
            for j in range(0, len(f)):
                y[i] += x[i - j] * f[-j - 1]
        return y

    @staticmethod
    def get_ema(x: np.ndarray, n: int) -> np.ndarray:
        """Apply an EMA."""
        x = x.copy()
        T = len(x)
        x['ema'] = 1
        x.iloc[0 : n - 1, 1] = np.nan
        x.iloc[n - 1, 1] = np.mean(x.iloc[0:n, 0])
        smoothing = 2 / (n + 1)
        for i in range(n, T):
            x.iloc[i, 1] = smoothing * x.iloc[i, 0] + (1 - smoothing) * x.iloc[i - 1, 1]
        return x
