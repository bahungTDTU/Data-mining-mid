$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force data/raw | Out-Null
$sources = @{
    'australian_user_reviews.json.gz' = 'https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz'
    'steam_games.json.gz' = 'https://cseweb.ucsd.edu/~wckang/steam_games.json.gz'
}
$expected = @{
    'australian_user_reviews.json.gz' = 'a934bf74d8579bb25bc7bf24e652a87e2ccc91064e429a4cb4af5809178687a8'
    'steam_games.json.gz' = '717f4653a26585fec8f463f59311a654127c2ed66b07b9f78b4c0ce64961c740'
}
foreach ($name in $sources.Keys) {
    $destination = Join-Path 'data/raw' $name
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination).Hash.ToLower() -ne $expected[$name]) {
            throw "Checksum mismatch: $destination. Preserve and inspect before replacing."
        }
        Write-Host "Verified $destination"
        continue
    }
    # Windows curl uses the OS trust store; do not disable certificate validation.
    & curl.exe --fail --location --continue-at - --retry 2 --max-time 300 --silent --show-error $sources[$name] -o "$destination.part"
    if ($LASTEXITCODE -ne 0) { throw "Download incomplete; rerun to resume $destination.part" }
    if ((Get-FileHash -LiteralPath "$destination.part").Hash.ToLower() -ne $expected[$name]) {
        throw "Downloaded checksum changed: $name. Investigate upstream; do not silently update pin."
    }
    Move-Item -LiteralPath "$destination.part" -Destination $destination
}
Get-FileHash data/raw/*.gz -Algorithm SHA256
