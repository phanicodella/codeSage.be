# Path: codeSage.be/run.py

import os
import sys
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import create_app
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main async function to initialize and run the application"""
    try:
        app = await create_app()
        return app
    except Exception as e:
        logging.critical(f"Failed to create application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        # Initialize the app
        app = asyncio.run(main())
        
        # Get configuration
        host = app.config.get('API_HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', app.config.get('API_PORT', 5000)))
        debug = app.config.get('DEBUG_MODE', False)
        
        # Run the Flask application
        app.run(
            host=host,
            port=port,
            debug=debug
        )
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        sys.exit(1)