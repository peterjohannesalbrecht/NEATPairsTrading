"""Module that support the Benchmarking.

This module holds the BenchmarkingService class which
provides multiple Services to the Benchmarking class.
"""
import pickle
import sqlite3
from typing import Optional

import neat
import numpy as np
import pandas as pd

from src.Benchmarking.BacktestingEnvironment import BacktestingEnvironment
from src.ResultService.VisualizationService import VisualizationService


class BenchmarkingService:
    """Class that supports the Benchmarking.

    This class provides multiple Services to the Benchmarking class.
    It computes backtests and provides information about it. Furthermore
    it computes multiple plots that visualize the results of the
    respective models and the benchmarking itself.
    """

    def __init__(self, data_test: pd.DataFrame) -> None:
        self.data_test = data_test
        self.env_neat = BacktestingEnvironment(self.data_test.copy(), 1000000)
        self.env_static = BacktestingEnvironment(self.data_test.copy(), 1000000)
        self.env_drl = BacktestingEnvironment(self.data_test.copy(), 1000000)
        self.config_neat = None
        self.network_neat = None
        self.stats_neat = None

    # ================= Services for the NEAT - Model =========================== #
    def run_backtest_neat(self, net: str, stats: str) -> None:
        """Run a backtest for the NEAT-model."""
        # Provide generated NEAT neural net and stats
        with open(net, 'rb') as f:
            self.network_neat = pickle.load(f)
        with open(stats, 'rb') as f:
            self.stats_neat = pickle.load(f)

        # Setup config
        self.config_neat = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            'src/Input/neat_config.txt',
        )

        # Perform backtest
        net = neat.nn.RecurrentNetwork.create(self.network_neat, self.config_neat)
        obs = self.env_neat.reset()
        done = False
        while not done:
            obs = obs.flatten()
            action_probabilities = net.activate(obs)
            action = np.argmax(action_probabilities) - 1
            obs, reward, done, info = self.env_neat.step(action)

    def generate_visuals_neat(self) -> None:
        """Generate plots for the NEAT-model.

        Create multiple plots that visualize the results
        of the NEAT model, including:
            - the generated neural network
            - the species over the training process
            - the statistics of the trainig process
        """
        viz = VisualizationService(
            self.env_neat.portfolio.get_portfolio_info(),
            self.data_test,
            env=self.env_neat,
        )
        viz.draw_net(self.config_neat, self.network_neat)
        viz.plot_species(self.stats_neat)
        viz.plot_stats(self.stats_neat)

    def plot_neat_portfolio_info(self, date_range: Optional[list] = None) -> None:
        """Plot information about the NEAT backtesting portfolio."""
        viz = VisualizationService(
            self.env_neat.portfolio.get_portfolio_info(), self.data_test, self.env_neat
        )
        viz.plot_exposure(compare='spread', wealth=True, date_range=date_range)
        viz.plot_exposure(compare='z_score', wealth=True, date_range=date_range)

    @staticmethod
    def compute_architecture_stats() -> tuple:
        """Compute average number of nodes/connections."""
        num_nodes = []
        num_connections = []
        query = 'SELECT * FROM final_pairs_mapping'

        with sqlite3.connect('src/Data/pairs_trading.db') as conn:
            mapping = pd.read_sql(query, conn)
        networks = list(mapping.pair)

        data_frames = []

        for network in networks:
            with open('src/Output/nets/net_'+ network + '.pkl', 'rb') as f:
                network_neat = pickle.load(f)
                num_nodes = len(network_neat.nodes.keys())
                num_connections = len(network_neat.connections.keys())
                df = pd.DataFrame({
                    'pair': [network],
                    '# hidden nodes': [num_nodes],
                    '# connections': [num_connections]
                })

                data_frames.append(df)  # Add the DataFrame to the list

        # Concatenate all DataFrames in the list together
        result_df = pd.concat(data_frames, ignore_index=True)
        # Subtract output nodes
        result_df['# hidden nodes'] = result_df['# hidden nodes'] - 3
        return result_df['# hidden nodes'].mean(), result_df['# connections'].mean()


    # ================= Services for the Static - Model =========================== #

    def run_backtest_static(self) -> None:
        """Run a backtest for the linear model."""
        # Perform backtest
        obs = self.env_static.reset()
        done = False
        action = 0
        while not done:
            obs = obs.flatten()
            if action == 1 and obs[0] >= 0.5:
                action = 0
                obs, reward, done, info = self.env_static.step(action)
                continue
            if action == -1 and obs[0] <= 0.5:
                action = 0
                obs, reward, done, info = self.env_static.step(action)
                continue
            if obs[0] > 0.7:
                action = -1
                obs, reward, done, info = self.env_static.step(action)
                continue
            if obs[0] < 0.3:
                action = 1
                obs, reward, done, info = self.env_static.step(action)
                continue
            if action not in [-1, 1]:
                action = 0
                obs, reward, done, info = self.env_static.step(action)
                continue
            else:
                action = action
                obs, reward, done, info = self.env_static.step(action)
                continue

    def plot_static_portfolio_info(self, date_range: Optional[list] = None) -> None:
        """Plot information about the linear backtesting portfolio."""
        viz = VisualizationService(
            self.env_static.portfolio.get_portfolio_info(),
            self.data_test.copy(),
            env=self.env_static,
        )
        viz.plot_exposure(compare='spread', wealth=True, date_range=date_range)
        viz.plot_exposure(compare='z_score', wealth=True, date_range=date_range)

    # ================= Services for the DRL - Model =========================== #

    def run_back_test_drl(self, model_non_linear) -> None:
        """Run a backtest for the non-linear model."""
        obs = self.env_drl.reset()
        done = False
        while not done:
            action, _states = model_non_linear.predict(obs)
            obs, rewards, done, info = self.env_drl.step(action)

    def plot_drl_portfolio_info(self, date_range: Optional[list] = None) -> None:
        """Plot information about the non-linear backtesting portfolio."""
        viz = VisualizationService(
            self.env_drl.portfolio.get_portfolio_info().copy(),
            self.data_test,
            self.env_drl,
        )
        viz.plot_exposure(compare='spread', wealth=True, date_range=date_range)
        viz.plot_exposure(compare='z_score', wealth=True, date_range=date_range)

    # ================= Services for Computing metrics ========================= #

    @staticmethod
    def compute_cagr(portfolio_info: pd.DataFrame) -> float:
        """Compute the annualized return of a portfolio."""
        # Backtesting ends if Wealth == 0, i.e. CAGR
        if portfolio_info['portfolio_value'].iloc[-1] < 0:
            # Approximate to be able to compute CAGR
            portfolio_info['portfolio_value'].iloc[-1] = 1
        portfolio_info = portfolio_info.reset_index()
        portfolio_info['Date'] = pd.to_datetime(portfolio_info['Date'])
        beginning_value = portfolio_info['portfolio_value'].iloc[0]
        ending_value = portfolio_info['portfolio_value'].iloc[-1]
        portfolio_info['Year'] = portfolio_info['Date'].dt.year
        num_days = portfolio_info.groupby('Year')['Date'].count().sum()
        num_years = num_days / 260

        # Calculate the annualized return
        return (ending_value / beginning_value) ** (1 / num_years) - 1

    @staticmethod
    def compute_sharpe_ratio(portfolio_info: pd.DataFrame) -> float:
        """Compute the annualized return of a portfolio.

        The risk-free rate is the geometric average of the
        FED-Fund-rate over the testing period.
        """
        returns = portfolio_info['portfolio_value'].pct_change()
        return (returns.mean() * 260 - 0.004) / (np.std(returns) * np.sqrt(260))

    @staticmethod
    def compute_number_of_trades(portfolio_info: pd.DataFrame) -> int:
        """Compute the number of executed trades."""
        exposure_diff = portfolio_info['exposure'].diff()
        # Count the number of times the 'Exposure' changes
        return (exposure_diff != 0).sum()

    # ================= Services for Summary metrics ========================= #

    @staticmethod
    def compute_summary_ranking(frame: pd.DataFrame) -> pd.DataFrame:
        """Compute a wealth ranking.

        This method computes a wealth ranking of the three methods.
        It indicates how many times each model was the best, second
        and third best model.
        """
        df = frame[['Wealth NEAT', 'Wealth STATIC', 'Wealth DRL']]
        ranking_df = pd.DataFrame(columns=['Best', 'Second', 'Third'])

        # Iterate through pairs
        for _, row in df.iterrows():
            # Get the column name of the best model (highest score) for this row
            best_model = row.idxmax()

            # Drop the best model from the row to find the second best
            row_without_best = row.drop(best_model)
            second_model = row_without_best.idxmax()

            # Drop the best and second best models to find the third best
            row_without_best_second = row.drop([best_model, second_model])
            third_model = row_without_best_second.idxmax()

            # Append the rankings to the ranking_df
            ranking_df = ranking_df.append({
                'Best': best_model,
                'Second': second_model,
                'Third': third_model
            }, ignore_index=True)

        # Count how many times each model was best, second, or third
        best_counts = ranking_df['Best'].value_counts()
        second_counts = ranking_df['Second'].value_counts()
        third_counts = ranking_df['Third'].value_counts()

        # Combine the counts into a new DataFrame
        return pd.DataFrame({
            'Best': best_counts,
            'Second': second_counts,
            'Third': third_counts
        }).fillna(0).astype(int)

    @staticmethod
    def ratio_of_winning_pairs(result_frame: pd.DataFrame) -> list:
        """Compute the ration of winning pairs.

        Computes the ratio of pairs in the backtesting that yielded an
        overall positive return, i.e. the Wealth is increased over the
        testing period.
        """
        ratios = []
        for model in ['NEAT', 'STATIC', 'DRL']:
            num_pairs = len(result_frame)
            num_winners= len(result_frame[result_frame[f'Wealth {model}'] >= 1e6])
            ratios.append(num_winners/num_pairs)
        return ratios

    def plot_wealth_comparison(self) -> None:
        """Plot wealth of all models."""
        VisualizationService.plot_wealth(
            self.env_neat.portfolio.get_portfolio_info(),
            self.env_static.portfolio.get_portfolio_info(),
            self.env_drl.portfolio.get_portfolio_info(),
        )
