# Use a base Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Download NLTK stopwords (as this is part of your code)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the application (you might want to add a specific command if this is a script)
CMD ["python", "TweetVibe Insights.py"]
