# Start the DR screening interface and open it in the browser.
#   .\run_app.ps1                 (or double-click run_app.bat)
#   .\run_app.ps1 -Port 7870 -DemoImage "C:\path\to\fundus.jpg"
# Python is taken from -Python, then $env:DR_PYTHON, then .venv in this folder, then the project's usual virtual environment, then PATH.
param([string]$Python = "", [int]$Port = 7860, [string]$DemoImage = "")

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidates = @(@($Python, $env:DR_PYTHON, (Join-Path $root ".venv\Scripts\python.exe"), "$env:USERPROFILE\venvs\dr\Scripts\python.exe") | Where-Object { $_ -and (Test-Path $_) })
$py = if ($candidates) { $candidates[0] } else { (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Host "Python not found. Install the requirements (pip install -r requirements.txt) or pass -Python." -ForegroundColor Red; exit 1 }

$argsList = @("-B", (Join-Path $root "app\app.py"), "--port", $Port)
if ($DemoImage) { $argsList += @("--demo-image", $DemoImage) }
Write-Host "Starting the interface with $py on port $Port ..." -ForegroundColor Cyan
$proc = Start-Process -FilePath $py -ArgumentList $argsList -WorkingDirectory $root -PassThru -NoNewWindow
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { break } } catch { Start-Sleep 2 }
}
Start-Process "http://127.0.0.1:$Port/"
Write-Host "Interface running at http://127.0.0.1:$Port/  (press Ctrl+C or close this window to stop)" -ForegroundColor Green
Wait-Process -Id $proc.Id
