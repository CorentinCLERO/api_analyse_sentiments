import os

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "sentiments_user"),
        password=os.getenv("MYSQL_PASSWORD", "sentiments_password"),
        database=os.getenv("MYSQL_DATABASE", "sentiments_db"),
    )


def fetch_tweets():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT text, positive, negative FROM tweets")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
