from __future__ import absolute_import
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("sndtrck")
logger.setLevel(logging.INFO)


def get_logger():
    return logger
