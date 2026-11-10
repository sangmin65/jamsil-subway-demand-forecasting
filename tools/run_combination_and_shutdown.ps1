# Explicit user request: shutdown only after approved fits, restore, report, commit and push succeed.
$ErrorActionPreference = 'Stop'
$repoDir = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repoDir
$jobDir = Join-Path $repoDir 'outputs/lstm-combination-control-2026-11-10'
New-Item -ItemType Directory -Force -Path $jobDir | Out-Null
$controlStatus = Join-Path $jobDir 'completion.json'
function Write-ControlStatus($phase, $detail) {
    @{phase=$phase; detail=$detail; time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $controlStatus -Encoding utf8
}
try {
    Write-ControlStatus 'training' 'EXP-012: three seeds; shutdown only after all checks.'
    & pwsh.exe -NoProfile -ExecutionPolicy Bypass -File tools/run_wsl_python.ps1 tools/run_lstm_combination.py --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/lstm-combination-2026-11-10
    if ($LASTEXITCODE -ne 0) { throw "Training/verification failed: $LASTEXITCODE" }
    $result = Get-Content -LiteralPath reports/metrics/lstm_combination.json -Raw -Encoding utf8 | ConvertFrom-Json
    if ($result.status -ne 'completed' -or $result.results.Count -ne 12 -or @($result.results | Where-Object { -not $_.reload_verified }).Count -ne 0) { throw 'Completion gates failed.' }
    $branch = (& git branch --show-current).Trim()
    if ($branch -ne 'feat/initial-experiment-setup') { throw 'Branch changed; do not commit or shutdown.' }
    & git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Diff check failed.' }
    $staged = & git diff --cached --name-only
    if ($staged) { throw 'Unexpected staged files; do not include them in automatic result commit.' }
    & git add -- reports/experiments/EXP-012_lstm-combination.md reports/metrics/lstm_combination.json
    if ($LASTEXITCODE -ne 0) { throw 'Staging results failed.' }
    & git commit -m 'experiment: LSTM 3층 배치8 조합 결과 비교 기록'
    if ($LASTEXITCODE -ne 0) { throw 'Result commit failed.' }
    $resultCommit = (& git rev-parse HEAD).Trim()
    & git push origin feat/initial-experiment-setup
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results committed locally; shutdown cancelled.' }
    Write-ControlStatus 'ready_to_shutdown' $resultCommit
    # Leave a cancellation window; /t 0 avoids the implicit force behavior of a nonzero shutdown timeout.
    Start-Sleep -Seconds 60
    if (Test-Path -LiteralPath (Join-Path $jobDir 'cancel_shutdown')) {
        Write-ControlStatus 'shutdown_cancelled' $resultCommit
        exit 0
    }
    Write-ControlStatus 'shutdown_requested' $resultCommit
    & shutdown.exe /s /t 0
    if ($LASTEXITCODE -ne 0) { throw "Windows shutdown failed: $LASTEXITCODE" }
} catch {
    Write-ControlStatus 'failed_no_shutdown' $_.Exception.Message
    Write-Error $_
    exit 1
}

