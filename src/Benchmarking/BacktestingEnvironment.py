"""Module for the Backtesting environment.

This Module holds a custom environment class for Backtesting
It inherits from the custom PairsTrading environment.
For benchmarking purposes it is also usable with the
stable-baselines3 package.
"""
import numpy as np
import pandas as pd

from src.PairsTrading.NeatEnvironment import NeatEnvironment


class BacktestingEnvironment(NeatEnvironment):
    """Class for a custom OpenAI Gym Environment.

    This class is defines a custom environment class for Backtesting.
    It inherits from the custom PairsTrading environment.
    For benchmarking purposes it is also usable with the
    stable-baselines3 package.
    """

    def __init__(self, data: pd.DataFrame, initial_cash: int) -> None:
        super().__init__(data, initial_cash)
        self.starting_step = 0

    def step(self, action: int) -> tuple:
        """Perform a step in the environment.

        This method performs a step in the environment
        by moving forward one step in time. It performs
        the action provided as an argument. A reward is
        received. The next observation is provided.
        If the end of the data has been reached the
        variable done will be set to True. This method
        overrides the method of the parent class.

        """
        self._take_action(action)
        self.current_step += 1
        reward = self.portfolio.get_latest_portfolio_return()
        done = (self.portfolio.get_portfolio_info().iloc[-1, 5]) <= 0
        if self.current_step == len(self.data):
            done = True
            return np.zeros((19,)), reward, done, {}
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.starting_step = 0
        self.current_step = self.starting_step
        self.portfolio.reset()
        return self._next_observation()
