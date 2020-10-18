FROM tiangolo/uvicorn-gunicorn:python3.7


ADD requirements.txt /app/
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

