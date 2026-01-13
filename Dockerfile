# 1. The Base Image: Start with a lightweight Python version
FROM python:3.9-slim

# 2. The Setup: Create a working directory inside the container
WORKDIR /app

# 3. The Dependencies: Copy requirements first (for caching speed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. The Code: Copy the rest of your app code
COPY . .

# 5. The Command: How to run the app?
# (If it's a simple script:)

# This tells Docker Desktop that the container listens on port 8501
EXPOSE 8501

# (OR: If it's a Streamlit web app, comment out the line above and use this instead:)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
