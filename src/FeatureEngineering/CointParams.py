"""This modules holds the CointParams()-class."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from filterpy.kalman import KalmanFilter


class CointParams:
    """Class handling computation of Cointegration Parameters.

    This class handles the computation of Cointegration parameters.
    Currently two methods are implemented for the computation: A
    rolling OLS and the Kalman Filter.
    """

    def __init__(self, data, use_logs=True) -> None:
        self.data = data.copy()
        self.use_logs = use_logs

    def get_params_ols(self, smoothing: bool = False) -> np.ndarray:
        """Compute the cointegration parameters.

        Method that computes the parameters gamma, mu of the
        cointegration relationship using the rolling OLS.
        """
        data = np.log(self.data.copy()) if self.use_logs else self.data.copy()
        T = len(data)
        window_size = 100

        # Init vectors
        mu = pd.Series(np.zeros(T))
        mu[:] = np.nan
        gamma = pd.Series(np.zeros(T))
        gamma[:] = np.nan

        # Get rolling OLS estimates
        for t in range(window_size, T, 10):
            regressand = sm.add_constant(data.iloc[:, 1])[t - window_size : t]
            regressor = data.iloc[t - window_size : t, 0]
            mod = sm.OLS(regressor, regressand)
            fitted_wls = mod.fit()
            mu.iloc[t] = fitted_wls.params[0]
            gamma.iloc[t] = fitted_wls.params[1]

        # Fill NA's
        mu = np.array(mu.ffill())
        gamma = np.array(gamma.ffill())

        # Smoothing
        if smoothing:
            mu = self.apply_conv_filter(mu, np.repeat(1 / 15, 15))
            gamma = self.apply_conv_filter(gamma, np.repeat(1 / 15, 15))

        return mu, gamma

    def get_params_kalman(self, smoothing: bool = False) -> np.ndarray:
        """Compute the cointegration parameters.

        Method that computes the parameters gamma, mu of the
        cointegration relationship using the Kalman-Filter.
        """
        data = np.log(self.data.copy()) if self.use_logs else self.data.copy()

        # Run OLS for initial position estimates
        regressors = sm.add_constant(data.iloc[0:208, 1])
        regressand = data.iloc[0:208, 0]
        mod = sm.OLS(regressand, regressors)
        fit = mod.fit()
        init_mu = fit.params[0]
        init_gamma = fit.params[1]

        # Init matricies
        gamma = pd.Series(np.zeros(len(data)))
        mu = pd.Series(np.zeros(len(data)))

        # Init kalman filter and set params
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([init_mu, init_gamma])
        kf.H = np.ones((1, 2))
        kf.H[0, 1] = data.iloc[0, 1]
        kf.P = np.diag([1e-7, 1e-7])
        kf.R = 1.3e-4
        kf.Q[0, 0] = 1e-5
        kf.Q[1, 1] = 1e-5
        kf.F = np.eye(2)
        measurements = np.array(data.iloc[:, 0])

        # Run Filter
        for t in range(0, len(data)):
            kf.predict()
            H = np.ones((1, 2))
            H[0, 1] = data.iloc[t, 1]
            kf.update(z=measurements[t], H=H)
            x = kf.x
            gamma[t] = x[1]
            mu[t] = x[0]
        mu = np.array(mu)
        gamma = np.array(gamma)
        # Apply smoothing if applicable
        if smoothing:
            mu = self.apply_conv_filter(mu, np.repeat(1 / 30, 30))
            gamma = self.apply_conv_filter(gamma, np.repeat(1 / 30, 30))

        return mu, gamma

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
