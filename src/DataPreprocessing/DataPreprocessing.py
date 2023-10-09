"""Module that holds the DataPreprocessing()-class.

This module holds the DataPreprocessing()-class that
is used to prepare the data for feature engineering.
"""
import sqlite3

import pandas as pd
from tqdm import tqdm


class DataPreprocessing:
    """Class that handles data preprocessing.

    This class handles the data preprocessing for the
    data that is later used to train and test the
    NEAT Pairs Trading Reinforcement Model and its
    benchmark models
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_pair_from_sql(
        asset_1: str, asset_2: str, table: pd.DataFrame='train_set_raw'
        ):
        """Load a pair (its features and price data) from the SQL-database."""
        query = f'SELECT [{asset_1}], [{asset_2}], Date FROM {table}'
        with sqlite3.connect('src/Data/pairs_trading.db') as conn:
            assets = pd.read_sql(query, conn)
            assets['Date'] = pd.to_datetime(assets['Date'])
            assets = assets.set_index('Date')
            return assets.rename(
                columns={asset_1: 'price_asset_1', asset_2: 'price_asset_2'}
            )

    @staticmethod
    def load_raw_data_from_xlsx(path: str) -> tuple:
        """Load the raw data.

        Takes the raw data, that was downloaded using
        Eikon Refinitiv Datastream. Perform a train-test
        split and store the data in separate tables in SQL
        Database.
        """
        # Specify sheet names
        sheet_names = [
            'oil_futures',
            'gasoline_futures',
            'heating_oil_futures',
            'oil_futures_new',
            'heating_gasoline_futures',
            'spark_spread_futures',
            'diverse_futures',
            'governement_bonds_1',
            'governement_bonds_2',
            'governement_bonds_3',
            'metals',
            'bond_indices',
            'emmission',
            'lumber',
            'coffee',
            'corn',
            'gold',
            'diverse',
        ]

        # store each sheet in individual dataframe
        dataframes = {}
        for sheet_name in tqdm(sheet_names):
            df = pd.read_excel(
                path, decimal=',', parse_dates=True, sheet_name=sheet_name
            ).set_index('Date')
            dataframes[sheet_name] = df

        # Create one dataframe with all data
        data = pd.DataFrame(columns=['Date']).set_index('Date')
        for frame in tqdm(dataframes.keys()):
            data = data.merge(
                dataframes[frame], left_index=True, right_index=True, how='outer'
            )
            duplicates = data.filter(like='_y', axis=1)
            data = data.drop(columns=duplicates.columns)
            data.columns = data.columns.str.replace('_x', '')

        # Split Training data and Testing data
        data_train = data[data.index < '2013-01-01']
        data_train = data_train[data_train.index >= '2009-01-02']
        data_test = data[data.index >= '2013-01-01']
        data_test = data_test[data_test.index < '2023-06-01']
        return data, data_train, data_test
