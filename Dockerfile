# Use the official Python image as a base image
FROM python:3.12-slim
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application files
COPY . .
# Expose the port Flask runs on
EXPOSE 8000
# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
# Command to run the application
CMD ["flask", "run"]