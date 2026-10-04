param([switch]$LocalPostgres, [int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv/Scripts/python.exe')) { throw 'Create .venv and install backend/requirements.lock first; see README.' }
& .venv/Scripts/python.exe scripts/setup_local.py
if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed' }
if (-not $LocalPostgres) {
    & docker compose up -d --wait db
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop and retry, or use -LocalPostgres with an existing PostgreSQL DATABASE_URL.' }
}
& .venv/Scripts/python.exe -m alembic -c backend/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { throw 'Migration failed' }
& .venv/Scripts/python.exe -m backend.seed
if ($LASTEXITCODE -ne 0) { throw 'Verified import failed; check M1 artifacts and MODEL_RUN_ID.' }
Push-Location frontend
try {
    & npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
} finally { Pop-Location }
Write-Host "Open http://127.0.0.1:$Port (Ctrl+C stops API/web; database remains in its volume)."
& .venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
