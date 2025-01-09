import logging

__all__ = [ 'logger' ]

logger: logging.Logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

fh: logging.FileHandler = logging.FileHandler('main.log', mode='w')
fh.setLevel(logging.DEBUG)

ch: logging.StreamHandler = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter: logging.Formatter = logging.Formatter('%(levelname)s - %(message)s') #%(asctime)s - %(name)s - 
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)