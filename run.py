# Path: codeSage.be/run.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(
        host=app.config['API_HOST'],
        port=app.config['API_PORT'],
        debug=app.config['DEBUG_MODE']
    )