import datetime as dt
import psycopg2
from psycopg2 import pool
import pandas as pd
import logging
import sys
from dal_stock_sql.util import Singleton, timedelta_to_postgres_interval

logger = logging.getLogger(__name__)


@Singleton
class Dal:
    def init(self, database='postgres', user='admin', password='admin', host='127.0.0.1', port='5432',
             ver_twsapi=10, maxconn=2):
        self.ver_twsapi = ver_twsapi
        # Establishing the connection
        logger.info('connecting to host %s and DB %s' % (host, database))
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            1, maxconn,  # minconn, maxconn
            user=user,
            password=password,
            host=host,
            port=port,
            database=database
        )

    def get_connection(self):
        try:
            conn = self.connection_pool.getconn()
        except Exception as e:
            raise e
        return conn

    def release_connection(self, connection):
        self.connection_pool.putconn(connection)

    def end(self):
        self.connection_pool.closeall()

    def get_all_symbol(self, dbtable: str, date_start: dt = None, date_end: dt = None):
        if None not in [date_start, date_end]:
            sql = '''SELECT * FROM %s WHERE date >= to_timestamp(%s) AND date <= to_timestamp(%s);''' % \
                  (dbtable, date_start.strftime('%s'), date_end.strftime('%s'))
        elif date_start == date_end:
            sql = '''SELECT * FROM %s ;''' % (dbtable)
            print(sql)
        elif date_end is None:
            sql = '''SELECT * FROM %s WHERE date >= to_timestamp(%s);''' % \
                  (dbtable, date_start.strftime('%s'))
        else:
            sql = '''SELECT * FROM %s WHERE date <= to_timestamp(%s);''' % \
                  (dbtable, date_end.strftime('%s'))
        # pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                df = pd.DataFrame(result, columns=columns)
                # Specify the types directly at DataFrame creation
                df = df.astype({
                    'date': 'datetime64[ns, UTC]',
                    'open': 'float',
                    'high': 'float',
                    'low': 'float',
                    'close': 'float',
                    'volume': 'int'
                })
        except Exception as e:
            raise e
        finally:
            self.release_connection(conn)
        # remove TZ otherwise pandas DateTimeIndex lookup in pandas do not work
        #df['date'] = df['date'].map(lambda t: pd.to_datetime(t.replace(tzinfo=None)).to_pydatetime())
        df = df.set_index(pd.DatetimeIndex(df['date']))
        # drop bar if no volume
        df = df[df.volume != -1]
        return df

    def get_one_symbol(self, dbtable: str, symbol: str, date_start: dt = None, date_end: dt = None):
        if ':' not in symbol:
            logger.error('Incorrect symbol %s' % symbol)
            return
            # raise ValueError
        if None not in [date_start, date_end]:
            sql = '''SELECT * FROM %s WHERE symbol = '%s' AND date >= to_timestamp(%s) AND date <= to_timestamp(%s);''' % \
                  (dbtable, symbol, date_start.strftime('%s'), date_end.strftime('%s'))
        elif date_start == date_end:
            sql = '''SELECT * FROM %s WHERE symbol = '%s';''' % (dbtable, symbol)
            print(sql)
        elif date_end is None:
            sql = '''SELECT * FROM %s WHERE symbol = '%s' AND date >= to_timestamp(%s);''' % \
                  (dbtable, symbol, date_start.strftime('%s'))
        else:
            sql = '''SELECT * FROM %s WHERE symbol = '%s' AND date <= to_timestamp(%s);''' % \
                  (dbtable, symbol, date_end.strftime('%s'))
        # pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy.
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                df = pd.DataFrame(result, columns=columns)
                # Specify the types directly at DataFrame creation
                df = df.astype({
                    'date': 'datetime64[ns, UTC]',
                    'open': 'float',
                    'high': 'float',
                    'low': 'float',
                    'close': 'float',
                    'volume': 'int'
                })
        except Exception as e:
            raise e
        finally:
            self.release_connection(conn)
        # remove TZ otherwise pandas DateTimeIndex lookup in pandas do not work
        #df['date'] = df['date'].map(lambda t: pd.to_datetime(t.replace(tzinfo=None)).to_pydatetime())
        df = df.set_index(pd.DatetimeIndex(df['date']))
        # drop bar if no volume
        df = df[df.volume != -1]
        return df
