FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project
COPY . .

# Expose API port
EXPOSE 9696

# Run Waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]
