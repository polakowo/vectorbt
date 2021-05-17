#!/bin/bash

curl -O https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/ || exit
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib
rm ta-lib-0.4.0-src.tar.gz