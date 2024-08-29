import logging
import yaml

def setup_logging(log_file='project.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message):
    print(message)
    logging.info(message)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class DataProcessingError(Exception):
    """Custom exception for errors during data processing."""
    pass
