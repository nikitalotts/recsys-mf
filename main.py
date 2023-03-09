import os
import logger
from logger import logger as log
from webapp.controller import app

if __name__ == "__main__":
    log.info('app start')
    app.run(host='127.0.0.1', port=5000, debug=False)
    log.info('app finished')
