@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -c "import loraforge, torch, transformers, peft, accelerate, safetensors" >nul 2>&1
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -e ".[train]"
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m loraforge serve %*
