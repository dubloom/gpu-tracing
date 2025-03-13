# Use Python base image
FROM python:3.13

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir django requests ddtrace

# Expose port
EXPOSE 8000

# Run Django server
CMD ["DD_SERVICE=django-cuda", "DD_ENV=dev", "DD_VERSION=0.1", "ddtrace-run", "python", "manage.py", "runserver", "0.0.0.0:8000"]