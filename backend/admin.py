"""Explicit local CLI promotion, never a public registration field."""
import argparse
from sqlalchemy import select
from backend.app.db import connect
from backend.app.models import User

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('email')
    args = parser.parse_args()
    engine, factory = connect()
    with factory() as db:
        user = db.scalar(select(User).where(User.email == args.email.casefold()))
        if not user:
            raise SystemExit('Register this account in the app first')
        user.role = 'admin'
        db.commit()
        print('Role updated for the specified account')
    engine.dispose()
