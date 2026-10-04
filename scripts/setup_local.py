"""Generate local DB secret once; never print or overwrite an existing .env."""
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
path = root / '.env'
if path.exists():
    print('Preserving existing .env')
else:
    password = secrets.token_hex(24)
    example = (root / '.env.example').read_text()
    with path.open('x', encoding='utf-8') as handle:
        handle.write(example.replace('replace-with-random-local-password', password))
    print('Created .env with random local DB secret (not displayed)')
