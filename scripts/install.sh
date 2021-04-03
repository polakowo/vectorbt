curl -O https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
pip install .[cov,docs] empyrical
pip install ray || true
pip install ta || true
pip install pandas_ta || true
pip install TA-Lib || true
pip install twine