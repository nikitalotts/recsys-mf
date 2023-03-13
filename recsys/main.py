"""
Main file that have to be run to start application
"""

import os
from logger import logger as log
from webapp.controller import app
from waitress import serve
from werkzeug.serving import run_simple
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound


# mode in which the application will be started
# possible values:
# prod - production mode
# dev - developer mode
mode = 'prod'

# application port number
# 5000 by default
port = int(os.environ.get('PORT', 5000))

if __name__ == "__main__":
    """Main function that start model on server"""
    log.info('app start')
    app = DispatcherMiddleware(NotFound(), {app.config['APPLICATION_ROOT']: app})
    if mode == 'dev':
        run_simple('0.0.0.0', port, app)
    elif mode == 'prod':
        serve(app, host='0.0.0.0', port=port)
    else:
        log.error(f'wrong server mode: {mode}')
        raise NotImplementedError('Wrong server mode!')
    log.info('app finished')

