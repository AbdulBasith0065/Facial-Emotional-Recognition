FROM python:3.9-slim

WORKDIR /app

# Install only the bare minimum system libraries for OpenCV
RUN yum install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Use Streamlit's command instead of just "python app.py"
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
