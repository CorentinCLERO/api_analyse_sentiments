import os

import pymysql
from pymysql.cursors import DictCursor

from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "sentiments_user"),
        password=os.getenv("MYSQL_PASSWORD", "sentiments_password"),
        database=os.getenv("MYSQL_DATABASE", "sentiments_db"),
        cursorclass=DictCursor,
    )


def fetch_tweets():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, text, positive, negative FROM tweets")
            return cursor.fetchall()
    finally:
        connection.close()
