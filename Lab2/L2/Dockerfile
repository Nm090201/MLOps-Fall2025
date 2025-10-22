FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

EXPOSE 4000

CMD ["sh", "-c", "python model_training.py && uvicorn main:app --host 0.0.0.0 --port 4000"]