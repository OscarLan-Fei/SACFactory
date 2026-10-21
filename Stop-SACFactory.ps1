$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot 'data/runtime/sacfactory_server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $serverId = [int](Get-Content -LiteralPath $pidFile)
    $running = Get-CimInstance Win32_Process -Filter "ProcessId = $serverId"
    $appScript = Join-Path $PSScriptRoot 'app.py'
    if ($running -and $running.CommandLine.Contains($appScript)) {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $serverId"
        foreach ($child in $children) {
            if ($child.CommandLine -and $child.CommandLine.Contains($appScript) -and $child.CommandLine.Contains('streamlit')) {
                Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $serverId
        Write-Output 'SACFactory stopped. All local experiment data are retained.'
    } else { Write-Output 'No matching SACFactory process is running.' }
}
