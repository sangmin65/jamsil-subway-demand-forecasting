# Pass Python arguments, for example: ./tools/run_wsl_python.ps1 -m unittest discover -s tests -v
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$linuxRepo = (& wsl.exe -d Ubuntu-24.04 -u aienge --exec wslpath -u $repo).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve repository path in WSL.' }
$ErrorActionPreference = 'Continue'
& wsl.exe -d Ubuntu-24.04 -u aienge --cd $linuxRepo --exec bash tools/run_wsl.sh @args
exit $LASTEXITCODE
