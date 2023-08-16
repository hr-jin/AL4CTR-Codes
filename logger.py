from __future__ import print_function
import logging
import sys

def set_logger():
  logger = logging.getLogger("tensorflow")
  if len(logger.handlers) == 1:
    logger.handlers = []
    logger.setLevel(logging.INFO)  

    formatter = logging.Formatter(
      "%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

  return logger
