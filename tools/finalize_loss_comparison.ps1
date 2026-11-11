# Complete the authorized result commit only after the runner and restoration checks succeed.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $repo
$statePath = Join-Path $repo 'outputs/loss-comparison-2026-11-11/status.json'
$completionPath = Join-Path $repo 'outputs/loss-comparison-2026-11-11/git-completion.json'
try {
    while ($true) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.status -eq 'failed') { throw "Training failed: $($state.error)" }
        if ($state.status -eq 'completed') { break }
        Start-Sleep -Seconds 30
    }
    if ($state.results.Count -ne 3 -or @($state.results | Where-Object { -not $_.reload_verified }).Count) {
        throw 'Missing completed or verified models'
    }
    if ((git branch --show-current).Trim() -ne 'feat/initial-experiment-setup') { throw 'Branch changed' }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Unexpected staged changes' }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Whitespace verification failed' }
    git add -- reports/experiments/EXP-013_loss-comparison.md reports/metrics/loss_comparison.json
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
    git commit -m 'experiment: LSTM MAE·Huber·MSE 비교 결과 기록'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    $commit = (git rev-parse HEAD).Trim()
    git push origin feat/initial-experiment-setup
    if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
    @{status='completed';commit=$commit;time=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath $completionPath -Encoding utf8
} catch {
    @{status='failed';error=$_.Exception.Message;time=(Get-Date -Format o)} | ConvertTo-Json | Set-Content -LiteralPath $completionPath -Encoding utf8
    throw
}
