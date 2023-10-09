"""Module that performs benchmarking.

This module creates the benchmarking_frame for the
training set and the testing set. Both frames contain
several metrics (PEV, TR, CAGR, ASR) across all models
for the respective dataset.
"""
import logging
import warnings

from src.Benchmarking.Benchmarking import Benchmarking

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

logger.info('Computing comparison frame for training data...')

comparison_frame_train_set = Benchmarking().create_comparison_frame(data='training')
comparison_frame_train_set.to_csv('src/Output/comparison_frame_training_set_.csv', index=False)
logger.info('Comparison frame for training data computed. Saved to .csv!')

logger.info('Computing comparison frame for test data...')
comparison_frame_test_set = Benchmarking().create_comparison_frame(data='test')
comparison_frame_test_set.to_csv('src/Output/comparison_frame_test_set_.csv', index=False)
logger.info('Comparison frame for test data computed. Saved to .csv!')
