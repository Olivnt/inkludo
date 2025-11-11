# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Create a startup script that runs both services
RUN echo '#!/bin/bash\n\
uvicorn realtime_server:app --host 0.0.0.0 --port 5050 &\n\
streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script

CMD ["/app/start.sh"]
