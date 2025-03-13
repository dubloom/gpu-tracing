# Use Python base image
FROM python:3.13

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask torch flask_sqlalchemy ddtrace

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["DD_SERVICE=django-cuda", "DD_ENV=dev", "DD_VERSION=0.1", "ddtrace-run", "flask", "run"]