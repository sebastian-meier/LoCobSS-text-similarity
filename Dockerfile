FROM tensorflow/tensorflow:latest-gpu

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python setup.py

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 app:app