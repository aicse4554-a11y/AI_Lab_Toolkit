FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements-dl.txt ./
# Use requirements.txt instead if you do not need the Deep Learning page (much smaller image).
RUN pip install --no-cache-dir -r requirements-dl.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
