# Use the base image of Python 3.10-slim
FROM python:3.10-slim
# Set the working directory
WORKDIR /project

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/project
# Copy the requirements file to the container
COPY ./requirements.txt /project/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install the additional requirements
RUN pip install -r /project/requirements.txt
# Install  libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copy the entire project directory to the container
COPY . /project
# Command to run the application (replace with your actual run comman
CMD ["python", "app/main.py"]