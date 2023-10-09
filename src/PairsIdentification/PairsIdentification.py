"""Module for the identification of suitable pairs.

This module holds the PairsIdentification class which
collects several methods that help to asses whether
a given pair of financial instruments is a promising
candidate for a pairs trading strategy.
"""

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from hurst import compute_Hc
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from src.DataPreprocessing.DataPreprocessing import DataPreprocessing


class PairsIdentification:
    """Handles Identification of suitable pairs.

    This class collects several methods that to asses
    whether a given pair of financial instruments is a promising
    candidate for a pairs trading strategy
    """

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        if data is None:
            return

        self.pair = data

        # Compute spread
        x = self.pair.price_asset_1
        y = self.pair.price_asset_2
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        self.spread = results.resid

        # Initialize measures
        self.coint_p_value = None
        self.hurst_coefficient = None
        self.average_crossing_time = None

    def compute_coint_p_value(self) -> None:
        """Compute the p-value of cointegration."""
        cointegration_result = coint(
            self.pair.price_asset_1, self.pair.price_asset_2, trend='c'
        )
        self.coint_p_value = cointegration_result[1]

    def compute_hurst_coefficient(self) -> None:
        """Compute the Hurst Coefficient."""
        Hurst, c, data = compute_Hc(self.spread, simplified=False)
        self.hurst_coefficient = Hurst

    def compute_ACT(self) -> None:
        """Compute the Average Crossing Time."""
        process = self.spread.copy()
        mean = np.mean(process)
        above_mean = process > mean
        crossing_indices = np.where(np.diff(above_mean.astype(int)) != 0)[0] + 1
        crossing_times = np.diff(crossing_indices)
        self.average_crossing_time = np.mean(crossing_times)

    def report(self, print: bool = False) -> tuple:
        """Return metrics of given pair and print to console (if enabled)."""
        self.compute_ACT()
        self.compute_coint_p_value()
        self.compute_hurst_coefficient()
        if print:
            print('Cointegration p-value ', self.coint_p_value)
            print('Hurst coefficient of spread ', self.hurst_coefficient)
            print('ACT of spread ', self.average_crossing_time)
        return (self.average_crossing_time, self.coint_p_value, self.hurst_coefficient)

    def reset(self) -> None:
        """Reset metrics attributes of instance."""
        self.coint_p_value = None
        self.hurst_coefficient = None
        self.average_crossing_time = None

    @staticmethod
    def generate_identification_stats(pairs_mapping: pd.DataFrame) -> tuple:
        """Generate identification metrics for each pair."""
        # Get pairs mapping
        with sqlite3.connect('src/Data/pairs_trading.db') as conn:
            pairs_mapping = pd.read_sql('SELECT * FROM pairs_mapping', conn)
        identification = pd.DataFrame(
            columns=[
                'date_min',
                'date_max',
                'na_count_asset_1',
                'na_count_asset_2',
                'act',
                'coint',
                'hurst',
            ]
        )
        # Generate statistics for each pair
        for pair in tqdm(range(0, len(pairs_mapping))):
            # Load pairs from database
            data = DataPreprocessing().load_pair_from_sql(
                pairs_mapping.loc[pair, 'asset_1'],
                pairs_mapping.loc[pair, 'asset_2'],
                table='train_set_raw',
            )

            # Drop NaN's at beginning and end of dataset
            first_idx = data.first_valid_index()
            last_idx = data.last_valid_index()
            data = data.loc[first_idx:last_idx]

            # Requiring at least 100 observations to compute statistics,
            if len(data) <= 100:
                temp = pd.DataFrame(
                    {
                        'pair': 'too short',
                        'date_min': 'too short',
                        'date_max': 'too short',
                        'na_count_asset_1': 'too short',
                        'na_count_asset_2': 'too short',
                        'act': 'too short',
                        'coint': 'too short',
                        'hurst': 'too short',
                        'area': pairs_mapping.loc[pair, 'area'],
                    },
                    index=range(0, 1),
                )
                identification = pd.concat([identification, temp])
                continue

            # If pairs has NaN's its is also not admitted
            if data.isna().any().any():
                na_count_asset_1 = data['price_asset_1'].isna().sum()
                na_count_asset_2 = data['price_asset_2'].isna().sum()
                temp = pd.DataFrame(
                    {
                        'pair': pairs_mapping.loc[pair, 'pair'],
                        'date_min': None,
                        'date_max': None,
                        'na_count_asset_1': na_count_asset_1,
                        'na_count_asset_2': na_count_asset_2,
                        'act': None,
                        'coint': None,
                        'hurst': None,
                        'area': pairs_mapping.loc[pair, 'area'],
                    },
                    index=range(0, 1),
                )
                identification = pd.concat([identification, temp])
                continue

            # If no NaN's and long enough, compute statistics
            date_min = data.index.min()
            date_max = data.index.max()
            na_count_asset_1 = data['price_asset_1'].isna().sum()
            na_count_asset_2 = data['price_asset_2'].isna().sum()
            act, coint, hurst = PairsIdentification(data).report()
            temp = pd.DataFrame(
                {
                    'pair': pairs_mapping.loc[pair, 'pair'],
                    'date_min': date_min,
                    'date_max': date_max,
                    'na_count_asset_1': na_count_asset_1,
                    'na_count_asset_2': na_count_asset_2,
                    'act': act,
                    'coint': coint,
                    'hurst': hurst,
                    'area': pairs_mapping.loc[pair, 'area'],
                },
                index=range(0, 1),
            )
            identification = pd.concat([identification, temp])

        # Store pairs that are selected
        selected = identification[
            (identification['act'] <= 2)
            | (identification['coint'] < 0.1)
            | (identification['hurst'] < 0.5)
        ]
        selected = selected[
            (selected['na_count_asset_1'] == 0)
            & (selected['na_count_asset_2'] == 0)
            & (selected['date_min'] == '2009-01-02')
        ]
        unselected = identification[
            ~(
                (identification['act'] <= 2)
                | (identification['coint'] < 0.1)
                | (identification['hurst'] < 0.5)
            )
        ].dropna()
        selected_pairs = list(selected.pair.unique())
        identification['selected'] = np.where(
            identification.pair.isin(selected_pairs), True,  False
        )
        return identification, selected, unselected
