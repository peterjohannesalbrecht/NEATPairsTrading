import logging
import sqlite3
import warnings

import pandas as pd

from src.DataPreprocessing.DataPreprocessing import DataPreprocessing
from src.FeatureEngineering.FeatureEngineering import FeatureEngineering
from src.PairsIdentification.PairsIdentification import PairsIdentification

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

PATH = 'src/Data/data_PJA.xlsx'

# Read in all time series data from the .xlsx-file
logger.info('Loading in data from .xslx and storing in SQL Database...')
raw_data, raw_data_train, raw_data_test = DataPreprocessing() \
                                          .load_raw_data_from_xlsx(PATH)

# Read in the pairs that go in the pair selection process
pairs_mapping = pd.read_excel(PATH, sheet_name='pairs')
codes_mapping = pd.read_excel(PATH, sheet_name='names')

# Save raw data and mapping to SQL-database
conn = sqlite3.connect('src/Data/pairs_trading.db')
cursor = conn.cursor()
raw_data_test.to_sql('test_set_raw', conn, if_exists='replace', index=True)
raw_data_train.to_sql('train_set_raw', conn, if_exists='replace', index=True)
pairs_mapping.to_sql('pairs_mapping', conn, if_exists='replace', index=False)
codes_mapping.to_sql('codes_mapping', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

# Select pairs that are eligible for pairs trading based on data properties
# (NaN-values, length, etc..) as well as statistical tests (Average crossing time,
# Cointegration-p-value, hurst coefficient)
logger.info('Computing Pairs identification metrics...')
identification, selected, unselected = PairsIdentification() \
                                       .generate_identification_stats(pairs_mapping)
conn = sqlite3.connect('src/Data/pairs_trading.db')
cursor = conn.cursor()
selected.to_sql('selected_pairs', conn, if_exists='replace', index=True)
identification.to_sql('identification', conn, if_exists='replace', index=True)
unselected.to_sql('unselected', conn, if_exists='replace', index=True)

# Perform sql-like inner join to only keep pairs mapping for selected pairs
logger.info('Performing pairs selection...')
final_pairs_mapping = pairs_mapping.merge(selected, on='pair', how='inner') \
                                   .drop(columns=['area_y'])

final_pairs_mapping.to_sql('final_pairs_mapping', conn, if_exists='replace', index=True)

# Perform feature engineering seperately for each pair
# (includes compute features and scaling)
logger.info('Performing feature engineering...')
train = FeatureEngineering.prepare_features(final_pairs_mapping, 'train')
test = FeatureEngineering.prepare_features(final_pairs_mapping, 'test')

# Save engineered features to SQL-database
conn = sqlite3.connect('src/Data/pairs_trading.db')
cursor = conn.cursor()
train.to_sql('training_pairs', conn, if_exists='replace', index=False)
test.to_sql('test_pairs', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
logger.info('Data-Pipeline successfully executed...')

