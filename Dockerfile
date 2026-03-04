FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .

EXPOSE 8800

CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8800"]
