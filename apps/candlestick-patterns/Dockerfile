FROM python:3.15-rc-alpine3.22

RUN apt-get -y update && apt-get -y install gcc curl make

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY assets ./assets
COPY app.py .

CMD ["python", "app.py"]