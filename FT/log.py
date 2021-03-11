import os
import logging

from datetime import datetime
from logging.handlers import RotatingFileHandler

def create_logger_cct(name='counthing'):
    if not os.path.exists('utils/'):
        os.makedirs('utils/')
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler('utils/'+str('CCT')+'_' + datetime.now().strftime("%Y-%m-%d")+'.log', maxBytes=0, backupCount=0, encoding="utf-8")
    logger = logging.getLogger('CCT')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: [CCT] %(message)s'))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
