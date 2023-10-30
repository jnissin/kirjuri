FROM python:3.10-slim

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/

# Use tini to handle process signals and then launch the Gunicorn server
ENTRYPOINT ["tini", "--"]
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]