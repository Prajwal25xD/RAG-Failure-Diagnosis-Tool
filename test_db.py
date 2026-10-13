
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

connection_url = (
    f"postgresql+psycopg://{username}:{password}"
    f"@{host}:{port}/{database}"
)

engine = create_engine(connection_url)

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), version()")
        )
        database_name, version = result.fetchone()

        print("PostgreSQL connection successful!")
        print("Database:", database_name)
        print("Version:", version)

except Exception as error:
    print("Connection failed:", error)

finally:
    engine.dispose()