"""This module holds the PairsTradingPortfolioClass.

This modules is for the PairsTradingPortfolio-class.
This class is used in the neat-algorithm to buy/sell
assets and to get information on the current portfolio
"""
import math

import pandas as pd


class PairsTradingPortfolio:
    """Class for a Pairs-Trading portfolio.

    This class holds information about the current portfolio status.
    It can buy and sell shares as well as close open positions.
    """

    def __init__(self, cash: int) -> None:
        self.portfolio_value = cash
        self.initial_portfolio_value = cash
        self.cash = cash
        self.shares_asset_1 = 0
        self.shares_asset_2 = 0
        self.exposure = 0
        self.portfolio_info = pd.DataFrame(
            columns=[
                'asset_1',
                'price_asset_1',
                'asset_2',
                'price_asset_2',
                'cash',
                'portfolio_value',
                'exposure',
                'action',
                'type',
                'Date',
            ]
        )

    def get_portfolio_value(self, price_asset_1: float, price_asset_2: float) -> float:
        """Get the balance of the portfolio."""
        self.portfolio_value = (
            self.shares_asset_1 * price_asset_1
            + self.shares_asset_2 * price_asset_2
            + self.cash
        )
        return self.portfolio_value

    def go_long(
        self,
        price_asset_1: float,
        price_asset_2: float,
        action: int,
        date: str,
        weights: pd.Series,
    ) -> None:
        """Perform the action of going long.

        This method performs the action of going long the Pairs Trading-
        portfolio, i.e. buy asset 1 using 50% of the portfolio value and
        sell short asset 2 (selling volume is 50% of portfolio value).
        """
        if self.exposure != 0:
            msg = 'Already invested - Close open positions first!'
            raise ValueError(msg)
        self.shares_asset_1 = math.ceil(weights.weight_1 * self.cash / price_asset_1)
        self.shares_asset_2 = -math.ceil(weights.weight_2 * self.cash / price_asset_2)
        if self.shares_asset_1 * self.shares_asset_2 > 0:
            msg = 'Shares held of both assets must have opposite sign!'
            raise ValueError(msg)

        self.cash = (
            self.cash
            - self.shares_asset_1 * price_asset_1
            - self.shares_asset_2 * price_asset_2
        )
        self.exposure = 1
        self.get_portfolio_value(price_asset_1, price_asset_2)
        receipt = [
            [
                self.shares_asset_1,
                price_asset_1,
                self.shares_asset_2,
                price_asset_2,
                self.cash,
                self.portfolio_value,
                self.exposure,
                action,
                'open',
                date,
            ]
        ]
        self.portfolio_info = pd.concat(
            [
                self.portfolio_info,
                pd.DataFrame(receipt, columns=self.portfolio_info.columns),
            ],
            ignore_index=True,
        )

    def go_short(
        self,
        price_asset_1: float,
        price_asset_2: float,
        action: int,
        date,
        weights: pd.Series,
    ) -> None:
        """Perform the action of going short.

        This method performs the action of going short the Pairs Trading-
        portfolio, i.e. short sell asset 1 using 50% of the portfolio value
        (selling volume is 50% of portfolio value) and buy asset 2.
        """
        if self.exposure != 0:
            msg = 'Already invested - Close open positions first!'
            raise ValueError(msg)
        self.shares_asset_1 = -math.ceil(weights.weight_1 * self.cash / price_asset_1)
        self.shares_asset_2 = math.ceil(weights.weight_2 * self.cash / price_asset_2)
        if self.shares_asset_1 * self.shares_asset_2 > 0:
            msg = 'Shares held of both assets must have opposite sign!'
            raise ValueError(msg)
        self.cash = (
            self.cash
            - self.shares_asset_2 * price_asset_2
            - self.shares_asset_1 * price_asset_1
        )
        self.exposure = -1
        self.get_portfolio_value(price_asset_1, price_asset_2)
        receipt = [
            [
                self.shares_asset_1,
                price_asset_1,
                self.shares_asset_2,
                price_asset_2,
                self.cash,
                self.portfolio_value,
                self.exposure,
                action,
                'open',
                date,
            ]
        ]
        self.portfolio_info = pd.concat(
            [
                self.portfolio_info,
                pd.DataFrame(receipt, columns=self.portfolio_info.columns),
            ],
            ignore_index=True,
        )

    def hold(
        self, price_asset_1: float, price_asset_2: float, action: int, date
    ) -> None:
        """Perform no trading.

        This method is called when the agent decides for action == 0,
        In this case the transaction DataFrame is updated because
        the portfolio value is needed for each time step, regardless
        of wether trading is happening or not.
        """
        self.get_portfolio_value(price_asset_1, price_asset_2)
        receipt = [
            [
                self.shares_asset_1,
                price_asset_1,
                self.shares_asset_2,
                price_asset_2,
                self.cash,
                self.portfolio_value,
                self.exposure,
                action,
                'hold',
                date,
            ]
        ]
        self.portfolio_info = pd.concat(
            [
                self.portfolio_info,
                pd.DataFrame(receipt, columns=self.portfolio_info.columns),
            ],
            ignore_index=True,
        )

    def close_positions(
        self,
        price_asset_1: float,
        price_asset_2: float,
        action: int,
        type_transaction: str,
        date,
    ):
        """Close open positions."""
        # If exposure is long -> Sell shares of asset 1 and
        # buy back shares of asset 2
        if self.exposure == 1:
            self.cash += (
                self.shares_asset_1 * price_asset_1
                + self.shares_asset_2 * price_asset_2
            )
            self.shares_asset_1 = 0
            self.shares_asset_2 = 0

        if self.exposure == 0:
            msg = 'No open Positions'
            raise ValueError(msg)
        # If exposure is short -> Sell shares of asset 2 and
        # buy back shares of asset 1
        if self.exposure == -1:
            self.cash += (
                self.shares_asset_2 * price_asset_2
                + self.shares_asset_1 * price_asset_1
            )
            self.shares_asset_1 = 0
            self.shares_asset_2 = 0

        self.exposure = 0
        self.get_portfolio_value(price_asset_1, price_asset_2)
        receipt = [
            [
                self.shares_asset_1,
                price_asset_1,
                self.shares_asset_2,
                price_asset_2,
                self.cash,
                self.portfolio_value,
                self.exposure,
                action,
                type_transaction,
                date,
            ]
        ]
        self.portfolio_info = pd.concat(
            [
                self.portfolio_info,
                pd.DataFrame(receipt, columns=self.portfolio_info.columns),
            ],
            ignore_index=True,
        )

    def reset(self) -> None:
        """Reset portfolio."""
        self.portfolio_value = self.initial_portfolio_value
        self.initial_portfolio_value = self.initial_portfolio_value
        self.cash = self.initial_portfolio_value
        self.shares_asset_1 = 0
        self.shares_asset_2 = 0
        self.exposure = 0
        self.portfolio_info = pd.DataFrame(
            columns=[
                'asset_1',
                'price_asset_1',
                'asset_2',
                'price_asset_2',
                'cash',
                'portfolio_value',
                'exposure',
                'action',
                'type',
                'Date',
            ]
        )

    def get_latest_portfolio_return(self) -> float:
        """Get the portfolio return.

        This method computes the portfolio return.
        """
        try:
            value_t = self.get_portfolio_info().iloc[-1, 5]
            value_t_1 = self.get_portfolio_info().iloc[-2, 5]
            return (value_t - value_t_1) / value_t_1

        except IndexError:
            return 0.0

    def get_portfolio_info(self) -> pd.DataFrame:
        """Getter for portfolio_info."""
        if len(self.portfolio_info) == 0:
            return pd.DataFrame([0, 0, 0, 0, 0, self.initial_portfolio_value]).transpose()

        # If positions are closed and opened for one step we get
        # two entries for that data. That would skew the computations.
        # Therefore return only the open positions as they are the relevant once.
        return self.portfolio_info[
            self.portfolio_info.type != 'close and open'
        ].set_index('Date', drop=True)
