# Base image — Python 3.11 slim version for smaller image size
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Create necessary directories inside container
RUN mkdir -p mlruns ml monitoring logs

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start FastAPI app using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]