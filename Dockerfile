FROM python:3.9-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port $PORT"]