"""Module that handles the Benchmarking.

This module holds the Benchmarking class. It handles
all the benchmarking workflow. More specifically it
creates a frame with comparison metrics across all pairs
and models. The models are the NEAT-algorithm itself, the
classic linear-method with static thresholds used in the
literature and a DRL methods, that is a Deep RL-model.
"""
import sqlite3

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

from src.Benchmarking.BenchmarkingService import BenchmarkingService


class Benchmarking:
    """Class that handles the benchmarking.

    It handles all the benchmarking workflow. More specifically it
    creates a frame with comparison metrics across all pairs
    and models. The models are the NEAT-algorithm itself, the
    classic linear-method with static thresholds used in the
    literature and a DRL method, that is a DQN RL-model.

    """

    def __init__(self) -> None:
        query = 'SELECT * FROM selected_pairs'
        with sqlite3.connect('src/Data/pairs_trading.db') as conn:
            self.selected_pairs = pd.read_sql(query, conn)

    def create_comparison_frame(self, data: str = 'training') -> pd.DataFrame:
        """Create a frame with comparison metrics.

        This method runs backtesting on all three models that are to be
        compared. In order to do this it needs to train the DQN-RL model first.
        It calls respective methods from the BenchmarkingService to perform tasks.
        It returns a frame where each row corresponds to one pair and each
        column indicates the metric. The following metrics are computed:
            - Wealth at the end of the testing period
            - Compound Annual Growth Rate (CAGR)
            - Annualized Sharpe Ratio (ASR)
            - Number of performed trades during the testing period
        """
        # Initialize empty performance metrics
        performance_metrics = pd.DataFrame(
            columns=[
                    'pair',
                    'Wealth NEAT',
                    'Wealth STATIC',
                    'Wealth DRL',
                    'CAGR NEAT',
                    'CAGR STATIC',
                    'CAGR DRL',
                    'ASR NEAT',
                    'ASR STATIC',
                    'ASR DRL',
                    '# Trades NEAT',
                    '# Trades STATIC',
                    '# Trades DRL'
            ]
        )

        # Define pairs to backtest (all that have been selected)
        job = list(range(0, len(self.selected_pairs)))

        # Compute metrics for each pair
        for pair in tqdm(job):
            # Load test data for backtesting
            pair_id = self.selected_pairs.pair[pair]
            query = f'SELECT * FROM {data}_pairs WHERE pair = "{pair_id}" ORDER BY Date'
            with sqlite3.connect('src/Data/pairs_trading.db') as conn:
                test = pd.read_sql(query, conn)

            # Provide generated NEAT neural net and stats
            network = f'src/Output/nets/net_{pair_id}.pkl'
            stats = f'src/Output/stats/stats_{pair_id}.pkl'

            # Provide trained DRL model
            model_drl = PPO.load(
            f'src/Benchmarking/trained_benchmark_models/model_{pair_id}'
        )
            # Perform backtesting
            benchmarking = BenchmarkingService(test)
            benchmarking.run_backtest_neat(network, stats)
            benchmarking.run_backtest_static()
            benchmarking.run_back_test_drl(model_drl)

            # Report metrics to performance_metrics frame
            portfolio_end_value_neat = (
                benchmarking.env_neat.portfolio.get_portfolio_info().iloc[-1, 5]
            )
            portfolio_end_value_static = (
                benchmarking.env_static.portfolio.get_portfolio_info().iloc[-1, 5]
            )
            portfolio_end_value_drl = (
                benchmarking.env_drl.portfolio.get_portfolio_info().iloc[-1, 5]
            )
            return_neat = benchmarking.compute_cagr(
                benchmarking.env_neat.portfolio.get_portfolio_info()
            )
            return_static = benchmarking.compute_cagr(
                benchmarking.env_static.portfolio.get_portfolio_info()
            )
            return_drl = benchmarking.compute_cagr(
                benchmarking.env_drl.portfolio.get_portfolio_info()
            )
            sharpe_neat = benchmarking.compute_sharpe_ratio(
                benchmarking.env_neat.portfolio.get_portfolio_info()
            )
            sharpe_static = benchmarking.compute_sharpe_ratio(
                benchmarking.env_static.portfolio.get_portfolio_info()
            )
            sharpe_drl = benchmarking.compute_sharpe_ratio(
                benchmarking.env_drl.portfolio.get_portfolio_info()
            )
            number_trades_neat = benchmarking.compute_number_of_trades(
                benchmarking.env_neat.portfolio.get_portfolio_info()
            )
            number_trades_static = benchmarking.compute_number_of_trades(
                benchmarking.env_static.portfolio.get_portfolio_info()
            )
            number_trades_drl = benchmarking.compute_number_of_trades(
                benchmarking.env_drl.portfolio.get_portfolio_info()
            )
            performance_metrics = performance_metrics.append(
                {
                    'pair': pair_id,
                    'Wealth NEAT': portfolio_end_value_neat,
                    'Wealth STATIC': portfolio_end_value_static,
                    'Wealth DRL': portfolio_end_value_drl,
                    'CAGR NEAT': return_neat,
                    'CAGR STATIC': return_static,
                    'CAGR DRL': return_drl,
                    'ASR NEAT': sharpe_neat,
                    'ASR STATIC': sharpe_static,
                    'ASR DRL': sharpe_drl,
                    '# Trades NEAT': number_trades_neat,
                    '# Trades STATIC': number_trades_static,
                    '# Trades DRL': number_trades_drl
                },
                ignore_index=True,
            )
        performance_metrics['TR NEAT'] = performance_metrics['Wealth NEAT'] \
                                                / 1000000 - 1
        performance_metrics['TR STATIC'] = performance_metrics['Wealth STATIC'] \
                                                / 1000000 - 1
        performance_metrics['TR DRL'] = performance_metrics['Wealth DRL'] \
                                                / 1000000 - 1

        # -Infs can occur whilst computing ASR if no trade is performed
        performance_metrics.replace(-np.inf, 0, inplace=True)
        return performance_metrics
