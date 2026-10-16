$ErrorActionPreference = 'Stop'
$projectPython = 'D:\anaconda3\envs\dl_gpu\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Required interpreter is missing: $projectPython"
}
Set-Location -LiteralPath $PSScriptRoot
& $projectPython -X utf8 server.py
