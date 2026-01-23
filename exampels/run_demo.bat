@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "PATH=%~dp0PortableGit\bin;%~dp0python_embedded;%PATH%"
set "TEMP=%~dp0temp"
set "TMP=%~dp0temp"
set "HF_HOME=%~dp0huggingface_cache"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
call "venv\Scripts\activate.bat"
echo.
echo Launching Borealis Gradio Demo...
echo Note: The first launch will download the model files, which may take some time.
echo.
python gradio_demo.py
pause
