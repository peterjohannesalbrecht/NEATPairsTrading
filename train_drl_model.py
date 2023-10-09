"""Module for training the PPO RL model.

This module handles the training of the DRL model.
Namely the training of the PPO RL-model for all pairs.
"""
import logging
import sqlite3
import warnings

import pandas as pd
from stable_baselines3 import PPO
from tqdm import tqdm

from src.PairsTrading.PPOEnvironment import PPOEnvironment

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    with sqlite3.connect('src/Data/pairs_trading.db') as conn:
        selected_pairs = pd.read_sql('SELECT * FROM selected_pairs', conn)
    logger.info(f'Starting training for {len(selected_pairs)} pairs...')
    logger.info('Depending on the hardware this might take several days...')
    logger.info('Consider training only one pair for testing purposes...')
    logger.info('To do so, redefine the job variable to be a list of one element...')
    job = list(range(0, len(selected_pairs)))
    for pair in tqdm(job):
        pair_id = selected_pairs.pair[pair]
        query = f'SELECT * FROM training_pairs WHERE pair = "{pair_id}" ORDER BY Date'
        with sqlite3.connect('src/Data/pairs_trading.db') as conn:
            train = pd.read_sql(query, conn).drop(columns='pair')
        train_env = PPOEnvironment(train, 1000000)
        model_non_linear = PPO('MlpPolicy', train_env, verbose=1)
        model_non_linear.learn(total_timesteps=200000)
        model_non_linear.save(f'src/Benchmarking/trained_benchmark_models/model_{pair_id}')
        logger.info(f'Finished training for pair {pair_id}!')
