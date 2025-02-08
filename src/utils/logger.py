import logging
import yaml

__all__ = [ 'logger' ]


def load_config(file_path):
    """
    Charge un fichier YAML et retourne un dictionnaire avec les valeurs.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config: dict = load_config('config.yaml')
log_file: str = config['logger']['LOG_FILE']
log_level: str = config['logger']['LOG_LEVEL']
log_format: str = config['logger']['LOG_FORMAT']

logger: logging.Logger = logging.getLogger('main')
logger.setLevel(log_level)

fh: logging.FileHandler = logging.FileHandler(log_file, mode='w')
fh.setLevel(log_level)

ch: logging.StreamHandler = logging.StreamHandler()
ch.setLevel(log_level)

formatter: logging.Formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)