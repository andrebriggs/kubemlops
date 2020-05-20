FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
# ADD MODEL_URL  ./
COPY score.py requirements.txt model.h5 ./

RUN pip install --no-cache-dir -r ./requirements.txt

CMD ["sh", "-c", "python score.py"]
