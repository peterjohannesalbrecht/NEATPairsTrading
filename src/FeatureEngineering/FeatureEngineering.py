"""Module for the Feature Engineering class.

Module that holds the FeatureEngineering class.
This class prepares features to be used for
the pairs trading algorithm.
"""
import pickle
from typing import Optional

import pandas as pd
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.DataPreprocessing.DataPreprocessing import DataPreprocessing
from src.FeatureEngineering.CointParams import CointParams
from src.FeatureEngineering.ZScore import ZScore
from src.PipelineSettings.PipelineSettings import PipelineSettings


class FeatureEngineering:
    """Class that handles the feature engineering.

    This class handles the feature engineering. To avoid
    data leakage/Look-ahead-Bias feature engineering is performed
    separately on training data and testing data.
    Features are scaled to the interval (0,1) for training. For
    testing data is transformed using the scaler object from the
    training scale-process.
    """

    def __init__(self, data: pd.DataFrame, pipeline_settings: PipelineSettings) -> None:
        self.data = data.copy()
        self.z_score = ZScore(data)
        self.coint_params = CointParams(data, use_logs=True)
        self.pipeline_settings = pipeline_settings

    def add_coint_params_features(self, method: str) -> None:
        """Add cointegration parameters as features."""
        if method == 'OLS':
            mu, gamma = self.coint_params.get_params_ols(smoothing=True)
        if method == 'Kalman':
            mu, gamma = self.coint_params.get_params_kalman(smoothing=True)
        self.data['mu'] = mu
        self.data['gamma'] = gamma
        self.data['weight_1'] = 1 / (1 + self.data.gamma)
        self.data['weight_2'] = self.data.gamma / (1 + self.data.gamma)

    def add_zscore_features(self, method: str, smoothing: bool) -> None:
        """Add several z_score features.

        Adds several z_score features, such as the z_score itself, but also
        moving averages, first differences, first lag, and functions of the
        former.
        """
        z_score_data = self.z_score.generate_z_score(method, smoothing)
        spread_data = self.z_score.spread.copy()
        self.data['z_score'] = z_score_data
        self.data['z_score_MA_50'] = self.data.loc[:, 'z_score'].rolling(50).mean()
        self.data['z_score_MA_50_current'] = (
            self.data.loc[:, 'z_score'] / self.data.loc[:, 'z_score_MA_50']
        )
        self.data['z_score_MA_25'] = self.data.loc[:, 'z_score'].rolling(25).mean()
        self.data['z_score_MA_25_current'] = (
            self.data.loc[:, 'z_score'] / self.data.loc[:, 'z_score_MA_25']
        )
        self.data['z_score_MA_10'] = self.data.loc[:, 'z_score'].rolling(10).mean()
        self.data['z_score_MA_10_current'] = (
            self.data.loc[:, 'z_score'] / self.data.loc[:, 'z_score_MA_10']
        )
        self.data['z_score_MA_5'] = self.data.loc[:, 'z_score'].rolling(5).mean()
        self.data['z_score_MA_5_current'] = (
            self.data.loc[:, 'z_score'] / self.data.loc[:, 'z_score_MA_5']
        )
        self.data['z_score_lagged'] = self.data.loc[:, 'z_score'].shift(1)
        self.data['z_score_differenced'] = self.data.loc[:, 'z_score'].diff(1)
        self.data['z_score_MSTD_10'] = self.data.loc[:, 'z_score'].rolling(10).std()
        self.data['spread'] = spread_data

    def add_bollinger_band_features(self) -> None:
        """Compute and add Bollinger Bands features."""
        # Format is required for FinTA package
        custom = self.data.copy()
        custom['high'] = None
        custom['low'] = None
        custom['open'] = None
        custom['close'] = custom['z_score']

        # Calculate BBANDS
        bbands = TA.BBANDS(custom)
        bbands.index = custom.index
        bbands['price'] = custom.close

        # Update the values based on the conditions
        bbands.loc[bbands['price'] > bbands['BB_UPPER'], 'signal'] = 1
        bbands.loc[bbands['price'] < bbands['BB_LOWER'], 'signal'] = -1
        bbands['signal'] = bbands.signal.fillna(0)

        # Append to data
        self.data['bb_upper'] = bbands.loc[:, 'BB_UPPER']
        self.data['bb_lower'] = bbands.loc[:, 'BB_UPPER']
        self.data['bb_signal'] = bbands.loc[:, 'signal']

    def add_MACD_features(self) -> None:
        """Compute and add MACD features."""
        # Format is required for FinTA package
        custom = self.data.copy()
        custom['high'] = None
        custom['low'] = None
        custom['open'] = None
        custom['close'] = custom['z_score']

        # Calculate MACD
        macd = TA.MACD(custom)
        macd['difference'] = macd.MACD - macd.SIGNAL

        # Append to data
        self.data['macd'] = macd.loc[:, 'MACD']
        self.data['macd_signal'] = macd.loc[:, 'SIGNAL']
        self.data['macd_signal_difference'] = macd.loc[:, 'difference']

    def add_RSI_features(self) -> None:
        """Compute and add RSI features."""
        # Format is required for FinTA package
        custom = self.data.copy()
        custom['high'] = None
        custom['low'] = None
        custom['open'] = None
        custom['close'] = custom['z_score']

        # Calculate RSI
        rsi = TA.RSI(custom, period=5)
        rsi.index = custom.index

        # Append to data
        self.data['rsi'] = rsi

    def scale_features(self,
                       features, scaler: MinMaxScaler = None,
                       pair: Optional[int] = None) -> tuple:
        """Return the final feature frame.

        This method returns the feature frame that can be used
        directly as observation frame in the NEAT algorithm.
        It selects the features to use and performs normalization/scaling
        to the interval (0,1).
        """
        features_to_use = self.pipeline_settings['features_to_use']
        data = features.loc[:, features_to_use]
        data = data.dropna().reset_index()
        df = pd.DataFrame(data)

        # Select the feature columns
        features = df[
            [
                col
                for col in df.columns
                if col
                not in [
                    'Date',
                    'price_asset_1',
                    'price_asset_2',
                    'weight_1',
                    'weight_2',
                    'pair',
                ]
            ]
        ]

        # Apply StandardScaler for normalization (training data)
        if scaler is None:
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(features)

            # Create a new DataFrame with normalized features
            df_normalized = pd.DataFrame(normalized_features, columns=features.columns)
            df_normalized['Date'] = df.Date
            df_normalized['price_asset_1'] = df.price_asset_1
            df_normalized['price_asset_2'] = df.price_asset_2
            df_normalized['weight_1'] = df.weight_1
            df_normalized['weight_2'] = df.weight_2
            df_normalized['pair'] = df.pair
            self.pipeline_settings['starting_time']
            output_dir = self.pipeline_settings['output_dir']
            with open(f'{output_dir}scalers/_scaler_{pair}.pkl', 'wb') as file:
                pickle.dump(scaler, file)
            return df_normalized, scaler

        # Apply already fitted Scaler for normalization (testing data)
        normalized_features = scaler.transform(features)

        # Create a new DataFrame with normalized features
        df_normalized = pd.DataFrame(normalized_features, columns=features.columns)
        df_normalized['Date'] = df.Date
        df_normalized['price_asset_1'] = df.price_asset_1
        df_normalized['price_asset_2'] = df.price_asset_2
        df_normalized['weight_1'] = df.weight_1
        df_normalized['weight_2'] = df.weight_2
        df_normalized['pair'] = df.pair
        return df_normalized, scaler

    def run_feature_engineering(
        self, params_method: str = 'Kalman', smoothing: bool = True
    ):
        """Runs feature-engineering pipeline for one pair."""
        self.add_zscore_features(params_method, smoothing)
        self.add_coint_params_features(params_method)
        self.add_bollinger_band_features()
        self.add_RSI_features()
        self.add_MACD_features()
        return self.data

    @staticmethod
    def prepare_features(final_pairs_mapping: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Perform feature-engineering for all pairs.

        The ``kind`` argument specifies if training data
        or testing data is prepared.
        """
        # Initialize empty feature frame
        all_pairs = pd.DataFrame()
        ps = PipelineSettings().load_settings()

        # Specify which data to use
        table = 'train_set_raw' if kind == 'train' else 'test_set_raw'

        # Loop over pairs to perform feature engineering
        for pair in tqdm(range(0, len(final_pairs_mapping))):
            data = DataPreprocessing().load_pair_from_sql(
                final_pairs_mapping.loc[pair, 'asset_1'],
                final_pairs_mapping.loc[pair, 'asset_2'],
                table=table,
            )

            # Remove NA's only at beginning or end of dataset
            first_idx, last_idx = data.first_valid_index(), data.last_valid_index()
            data = data.loc[first_idx:last_idx]

            # Run feature engineering for pair of current iteration
            fs = FeatureEngineering(data, ps)
            features = fs.run_feature_engineering(
                params_method='Kalman', smoothing=True
            )
            features['pair'] = final_pairs_mapping.pair[pair]

            if kind == 'test':
                # If features are engineered for testing, use scaler from training
                with open(f'src/Output/scalers/_scaler_{pair}.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            else:
                # If features are engineered for training, scale to (0,1)
                scaler = None
            features = fs.scale_features(features, pair=pair, scaler=scaler)[0]

            # Remove the first 208 observations,
            # because the initial_parameter estimates are computed over a horizon
            # of 208 trading days, which would induce Look-ahead Bias.
            if kind == 'test':
                features = features[features['Date'] >= '2013-10-19']
            else:
                features = features[features['Date'] >= '2009-10-22']
            all_pairs = pd.concat([all_pairs, features])

        return all_pairs
