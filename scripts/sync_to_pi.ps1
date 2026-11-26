# Copies this project from the PC to the Pi (~/assistant). Keeps data/ and config.toml on the Pi.
# Usage (PowerShell, from the project folder):  .\scripts\sync_to_pi.ps1  [-PiHost shadowlessblu@<ip>]
# The default "pi5" is the SSH alias in ~/.ssh/config for the Pi's Tailscale address (100.78.110.4).
param([string]$PiHost = "pi5")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$archive = Join-Path $env:TEMP "pi5-assistant.tgz"

Push-Location $root
try {
    # Use Windows' own bsdtar: Git's MSYS tar mangles C:\ temp paths.
    & "$env:SystemRoot\System32\tar.exe" --exclude=__pycache__ --exclude=.venv --exclude=.venv-research `
        --exclude=data --exclude=config.toml --exclude=experiments/results `
        --exclude=research/models --exclude=research/train.log -czf $archive `
        README.md HANDOVER.md pyproject.toml config.example.toml .gitignore `
        src tests scripts bench experiments prompts deploy firmware research
} finally { Pop-Location }

Write-Host "Copying to $PiHost ..."
scp $archive "${PiHost}:/tmp/pi5-assistant.tgz"
ssh $PiHost "mkdir -p ~/assistant && tar -xzf /tmp/pi5-assistant.tgz -C ~/assistant && rm /tmp/pi5-assistant.tgz && echo SYNC-OK && ls ~/assistant"
Remove-Item $archive
