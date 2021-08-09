FROM python:3.8-slim

RUN apt-get -y update && apt-get -y install gcc curl make

RUN pip install --upgrade pip
# Required by TA-Lib and numba
RUN pip install numpy>=1.19.4

RUN curl -O https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd ..

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY assets ./assets
COPY app.py .

CMD ["python", "app.py"]