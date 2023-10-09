"""Module for the PairsTrading Gym environment (Training).

This Module holds a custom environment class for PairsTrading.
It is used to train an algorithm for pairs trading using the
neat-python package. For benchmarking purposes it is also usable
with the stable-baselines3 package.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.PairsTrading.Portfolio import Portfolio


class NeatEnvironment(gym.Env):
    """Class for a custom OpenAI Gym Environment.

    This class is defines a custom OpenAI Gym Environment. It is
    used to build a PairsTrading-strategy using the neat-python
    package. Furthermore it is usable with the stable-baselines3
    package. It inherits from the boilerplate OpenAI Gym
    Environment class
    """

    def __init__(self, data: pd.DataFrame, initial_cash: int) -> None:
        self.data = data
        self.starting_step = 1
        self.current_step = self.starting_step
        self.portfolio = Portfolio(initial_cash)

        # Action space with actions: {-1, 0, 1}
        self.action_space = spaces.Discrete(3, start=-1)

        # Observation space (Required for stable-baseline)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1] - 6,), dtype=np.float64
        )
        # For plotting the network in Vizualisation Service
        self.input_nodes = [
            col
            for col in self.data.columns
            if col
            not in [
                'Date',
                'price_asset_1',
                'price_asset_2',
                'weight_1',
                'weight_2',
                'pair',
                'spread',
            ]
        ]

    def step(self, action: int) -> tuple:
        """Perform a step in the environment.

        This method performs a step in the environment
        by moving forward one step in time. It performs
        the action provided as an argument. A reward is
        received. The next observation is provided.
        If the end of the data has been reached the
        variable done will be set to True.
        """
        self._take_action(action)
        self.current_step += 1
        reward = self.portfolio.get_latest_portfolio_return()
        done = bool((self.portfolio.get_portfolio_info().iloc[-1, 5]) <= 0)
        if self.current_step == len(self.data):
            done = True
            return self.reset()[0], reward, done, {}
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        self.current_step = self.starting_step
        self.portfolio.reset()
        return self._next_observation()

    def _next_observation(self) -> np.ndarray:
        """Provide the next observations by getting the next row of the data."""
        obs = np.array(
            self.data.loc[
                self.current_step,
                [
                    col
                    for col in self.data.columns
                    if col
                    not in [
                        'Date',
                        'price_asset_1',
                        'price_asset_2',
                        'weight_1',
                        'weight_2',
                        'pair',
                        'spread',
                    ]
                ],
            ]
        ).flatten()
        return np.array(obs, dtype=np.float64)

    def _take_action(self, action: int) -> None:
        """Take a provided action.

        Interacts with the Portfolio to perform the provided
        trading action.
        """
        # Get current prices
        price_asset_1 = self.data.loc[self.current_step, 'price_asset_1']
        price_asset_2 = self.data.loc[self.current_step, 'price_asset_2']
        date = self.data.loc[self.current_step, 'Date']
        weights = self.data.loc[self.current_step, ['weight_1', 'weight_2']]

        if action == 1:
            # If already long, don't do anything
            if self.portfolio.exposure == 1:
                self.portfolio.hold(price_asset_1, price_asset_2, action, date)
                return

            # If short, close open position by buying back and go long
            if self.portfolio.exposure == -1:
                self.portfolio.close_positions(
                    price_asset_1, price_asset_2, action, 'close and open', date
                )
                self.portfolio.go_long(
                    price_asset_1, price_asset_2, action, date, weights
                )
                return

            # If no open position, buy portfolio
            # Go long 50% of balance in asset 1
            self.portfolio.go_long(price_asset_1, price_asset_2, action, date, weights)

        if action == 0:
            # If long or short -> close positions
            if self.portfolio.exposure != 0:
                self.portfolio.close_positions(
                    price_asset_1, price_asset_2, action, 'close', date
                )
                return
            # Else don't do anything
            self.portfolio.hold(price_asset_1, price_asset_2, action, date)

        if action == -1:
            # If already short, don't do anything
            if self.portfolio.exposure == -1:
                self.portfolio.hold(price_asset_1, price_asset_2, action, date)
                return

            # If long, close open position by selling and go short
            if self.portfolio.exposure == 1:
                self.portfolio.close_positions(
                    price_asset_1, price_asset_2, action, 'close and open', date
                )
                self.portfolio.go_short(
                    price_asset_1, price_asset_2, action, date, weights
                )
                return

            # If no open position, buy portfolio
            # Go long 50% of balance in asset 1
            self.portfolio.go_short(price_asset_1, price_asset_2, action, date, weights)
