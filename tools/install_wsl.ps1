# Install WSL2/Ubuntu without restarting the computer automatically.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $repo 'outputs/gpu-setup'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in an administrator PowerShell window.'
}
Start-Transcript -Path (Join-Path $logDir 'wsl-install.log') -Append
try {
    & wsl.exe --install -d Ubuntu-24.04 --no-launch
    $result = $LASTEXITCODE
    @{ exit_code = $result; recorded_at = (Get-Date).ToString('o'); note = 'Check installation output for restart requirement; no restart was requested by this script.' } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'wsl-install-result.json') -Encoding UTF8
    if ($result -ne 0) { throw "WSL installation exited with $result; see log." }
} finally {
    Stop-Transcript
}
