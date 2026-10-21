param([int]$Port = 8517, [switch]$Restart)
$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$taskRuntime = Join-Path $appRoot '.venv/Scripts/python.exe'
$appScript = Join-Path $appRoot 'app.py'
$logRoot = Join-Path $appRoot 'data/runtime'
if (!(Test-Path -LiteralPath $taskRuntime)) {
    throw 'Python environment missing. Create a virtual environment, install requirements.txt, then run: python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8517'
}
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$pidFile = Join-Path $logRoot 'sacfactory_server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $serverId = [int](Get-Content -LiteralPath $pidFile)
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $serverId"
    if ($running -and $running.CommandLine.Contains($appScript)) {
        if ($Restart) {
            & (Join-Path $appRoot 'Stop-SACFactory.ps1')
        } else {
            try {
                $health = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 3
                if ($health.Content.Trim() -eq 'ok') {
                    Write-Output "SACFactory already running: http://127.0.0.1:$Port. Use -Restart to reload updated modules."
                    exit 0
                }
            } catch { Write-Output 'Restarting an unresponsive SACFactory process.' }
            & (Join-Path $appRoot 'Stop-SACFactory.ps1')
        }
    }
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port $Port is occupied; use another -Port value." }
$arguments = @('-X','utf8','-m','streamlit','run',('"' + $appScript + '"'),'--server.address','127.0.0.1','--server.port',"$Port",'--server.headless','true','--server.runOnSave','true','--browser.gatherUsageStats','false')
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$process = Start-Process -FilePath $taskRuntime -ArgumentList $arguments -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'sacfactory_server.stdout.log') -RedirectStandardError (Join-Path $logRoot 'sacfactory_server.stderr.log')
Set-Content -LiteralPath $pidFile -Value $process.Id
Write-Output "SACFactory PID $($process.Id): http://127.0.0.1:$Port"
