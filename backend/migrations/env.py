from alembic import context
from sqlalchemy import create_engine
from backend.app.config import database_url
from backend.app.models import Base

def migrate(connection):
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()

provided = context.config.attributes.get('connection')
if provided is not None:
    migrate(provided)
else:
    engine = create_engine(database_url())
    with engine.connect() as connection:
        migrate(connection)
