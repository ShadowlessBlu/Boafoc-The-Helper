# Copy the Pi's newest assistant backup to this PC (keeps the newest 8). Runs weekly from Task Scheduler
# ("Pi5 assistant backup pull", Sundays 13:00, or at the next logon if the PC was off then).
#   powershell -ExecutionPolicy Bypass -File scripts\pull_backup.ps1            copy now
#   powershell -ExecutionPolicy Bypass -File scripts\pull_backup.ps1 -Install   create the weekly task
# The backups hold config.toml, which has the API token and MQTT password: keep this folder private.
param([switch]$Install, [string]$Dest = "$env:USERPROFILE\Backups\Pi5-AI-Assistant", [int]$Keep = 8)

$ErrorActionPreference = "Stop"
$log = Join-Path $Dest "pull.log"

if ($Install) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`""
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 13:00
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
    Register-ScheduledTask -TaskName "Pi5 assistant backup pull" -Action $action -Trigger $trigger `
        -Settings $settings -Description "Copy the Pi assistant's newest nightly backup to this PC" -Force | Out-Null
    Get-ScheduledTask -TaskName "Pi5 assistant backup pull" | Select-Object TaskName, State
    return
}

New-Item -ItemType Directory -Force $Dest | Out-Null
function Write-Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" | Add-Content $log }
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"
try {
    $name = (& $ssh -o BatchMode=yes -o ConnectTimeout=20 pi5 "ls -1t backups/assistant/assistant-*.tgz | head -1").Trim()
    if (-not $name) { throw "no backup on the Pi" }
    $file = Split-Path $name -Leaf
    $target = Join-Path $Dest $file
    if (Test-Path $target) { Write-Log "already have $file"; return }
    & $scp -q -o BatchMode=yes "pi5:$name" "$target.part"
    if ($LASTEXITCODE -ne 0) { throw "scp failed ($LASTEXITCODE)" }
    $remote = [long](& $ssh -o BatchMode=yes pi5 "stat -c %s $name")
    if ((Get-Item "$target.part").Length -ne $remote) { throw "size differs from the Pi's copy" }
    Move-Item "$target.part" $target -Force
    Get-ChildItem $Dest -Filter "assistant-*.tgz" | Sort-Object Name -Descending | Select-Object -Skip $Keep |
        Remove-Item -Force
    Write-Log "copied $file ($remote bytes)"
} catch {
    Write-Log "FAILED: $_"
    exit 1
}
