# Path: codeSage.be/docker-entrypoint.sh

#!/bin/bash
set -e

# Run database migrations
python -c "from src.models.migrations import init_db; init_db()"

# Start the Flask application with gunicorn
exec gunicorn --bind 0.0.0.0:5000 \
    --workers 4 \
    --threads 4 \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    "src.app:create_app()"