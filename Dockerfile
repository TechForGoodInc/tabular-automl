# Pull base image
FROM python:3.8

# Metadata
LABEL "maintainer" "Harison Gachuru <harisongachuru@gmail.com>"
LABEL "repository" "https://github.com/harisonmg/tabular-automl"
LABEL "homepage" "https://github.com/harisonmg/tabular-automl"

# Update apt repositories
# RUN apt-get update && apt-get install -y libgomp1

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code
RUN pip install -r requirements.txt

# Copy project
COPY . /code

# Run command
CMD ["python", "-m", "streamlit", "run", "web_app/app.py"]
