FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --workers 4 --bind :${PORT:-8080} app:app