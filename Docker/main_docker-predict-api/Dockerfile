FROM python:3.8-slim

WORKDIR /predict-api-test/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ /predict-api-test/

CMD [ "fastapi", "run", "app/main.py", "--port", "80" ]
