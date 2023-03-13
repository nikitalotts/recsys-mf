import logging
import os.path

LOGGING_LEVEL = logging.INFO
LOG_FILE_NAME = 'logs.log'
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE_NAME)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOGGING_LEVEL)

logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(levelname)s::%(asctime)s::%(module)s::%(funcName)s::%(filename)s::%(lineno)d %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_NAME, mode='a'), stream_handler],
                    datefmt='%d-%b-%y %H:%M:%S'
                    )


logger = logging.getLogger(__name__)
