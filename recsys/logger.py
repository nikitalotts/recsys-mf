import logging
import os.path

LOGGING_LEVEL = logging.INFO
LOG_FILE_NAME = 'logs.log'
LOG_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
LOG_FILE_PATH = os.path.join(LOG_FOLDER_PATH, LOG_FILE_NAME)

if not os.path.exists(LOG_FOLDER_PATH):
    os.makedirs(LOG_FOLDER_PATH)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOGGING_LEVEL)

logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='a'), stream_handler],
                    datefmt='%d-%b-%y %H:%M:%S'
                    )


logger = logging.getLogger(__name__)

