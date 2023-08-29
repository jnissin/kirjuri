FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y tini git
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/transcribe_test.py /app/

# Run app when the container launches, wrap with tini
ENTRYPOINT ["tini", "--"]
CMD ["python", "./transcribe_test.py"]