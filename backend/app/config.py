import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')


def database_url():
    url = os.environ.get('DATABASE_URL')
    if not url or not url.startswith('postgresql+psycopg://'):
        raise RuntimeError('Set PostgreSQL DATABASE_URL in .env; see README')
    return url


def origins():
    return os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000').split(',')
