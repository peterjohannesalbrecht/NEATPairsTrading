"""Module for training the NEAT model.

This module handles the training of the NEAT model.
Namely the training of the NEAT-model for all pairs.
"""
import logging
import pickle
import sqlite3
import warnings

import pandas as pd
from tqdm import tqdm

from src.PipelineSettings.PipelineSettings import PipelineSettings
from src.Training.Training import Training

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    with sqlite3.connect('src/Data/pairs_trading.db') as conn:
        selected_pairs =  pd.read_sql('SELECT * FROM selected_pairs', conn)
    settings = PipelineSettings().load_settings()
    logger.info(f'Starting training for {len(selected_pairs)} pairs...')
    logger.info('Depending on the hardware this might take several days...')
    logger.info('Consider training only one pair for testing purposes...')
    logger.info('To do so, redefine the job variable to be a list of one element')
    job = list(range(0,len(selected_pairs)))
    for pair in tqdm(job):
        pair_id = selected_pairs.pair[pair]
        settings['pair'] = pair_id
        net, stats = Training(settings).run_training()
        # Save the best network to a file
        time = settings['starting_time']
        output_dir = settings['output_dir']
        pair_id = selected_pairs.pair[pair]
        with open(f'{output_dir}net_"{pair_id}".pkl', 'wb') as f:
            pickle.dump(net, f)

        # Save the stats to a file
        with open(f'{output_dir}stats/stats_"{pair_id}".pkl', 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f'Finished training for pair {pair_id}!')
