# Use official Python base image with a specific version (recommended)
FROM python:3.9-slim

# Set working directory
WORKDIR /app



# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY base_iris_lab1.py test.py app.py client.py iris_extended_encoded.csv ./

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production 
ENV PORT=4000

# Expose the API port
EXPOSE 4000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]
