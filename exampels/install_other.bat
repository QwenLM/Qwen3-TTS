@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ===========================================================================
:: Borealis Demo Portable Installer
:: ===========================================================================

set "INSTALL_DIR=%~dp0"
set "PYTHON_DIR=%INSTALL_DIR%python_embedded"
set "VENV_DIR=%INSTALL_DIR%venv"
set "GIT_DIR=%INSTALL_DIR%PortableGit"

set "PYTHON_URL=https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/PortableGit-2.44.0-64-bit.7z.exe"

:: Setup local directories for cache and temp files
set "HF_HOME=%INSTALL_DIR%huggingface_cache"
set "PIP_CACHE_DIR=%INSTALL_DIR%pip_cache"
set "TEMP=%INSTALL_DIR%temp"
set "TMP=%INSTALL_DIR%temp"

set "MAX_RETRIES=3"
set "RETRY_DELAY=5"

echo.
echo -------------------------------------------------------------------------
echo     Borealis Demo Portable Installer
echo -------------------------------------------------------------------------
echo.
echo This script will install Borealis Demo in a fully portable mode into:
echo %INSTALL_DIR%
echo.
pause

:: Step 1: Clean old installation if exists
echo.
echo [1/6] Preparing for installation: cleaning up old files...
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
if exist "%HF_HOME%" rmdir /s /q "%HF_HOME%"
if exist "%PIP_CACHE_DIR%" rmdir /s /q "%PIP_CACHE_DIR%"
if exist "%TEMP%" rmdir /s /q "%TEMP%"
if exist "%GIT_DIR%" rmdir /s /q "%GIT_DIR%"
if exist "%PYTHON_DIR%" rmdir /s /q "%PYTHON_DIR%"
del "%INSTALL_DIR%run_demo.bat" 2>nul
mkdir "%TEMP%"
echo Done.

:: Step 2: Download and install portable Git
echo.
echo [2/6] Downloading and installing portable Git...
if not exist "%GIT_DIR%" (
    call :DownloadFile "%GIT_URL%" "%INSTALL_DIR%PortableGit.exe"
    if !errorlevel! neq 0 (
        echo Error: Failed to download Git after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    "%INSTALL_DIR%PortableGit.exe" -y -gm2 -nr -o"%GIT_DIR%"
    del "%INSTALL_DIR%PortableGit.exe"
)
set "PATH=%GIT_DIR%\bin;%PATH%"
echo Done.

:: Step 3: Download and extract portable Python
echo.
echo [3/6] Downloading and setting up portable Python 3.11...
if not exist "%PYTHON_DIR%" (
    mkdir "%PYTHON_DIR%"
    call :DownloadFile "%PYTHON_URL%" "%INSTALL_DIR%python.zip"
    if !errorlevel! neq 0 (
        echo Error: Failed to download Python after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%INSTALL_DIR%python.zip' -DestinationPath '%PYTHON_DIR%'"
    del "%INSTALL_DIR%python.zip"

    echo.
    echo Configuring Python to use pip...
    powershell -Command "(Get-Content '%PYTHON_DIR%\python311._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python311._pth'"
    
    echo.
    echo Installing pip...
    call :DownloadFile "%GET_PIP_URL%" "%PYTHON_DIR%\get-pip.py"
    if !errorlevel! neq 0 (
        echo Error: Failed to download get-pip.py after %MAX_RETRIES% attempts.
        pause
        exit /b 1
    )
    "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
)
echo Done.

:: Step 4: Install virtualenv package
echo.
echo [4/6] Installing virtualenv...
call :RunCommandWithRetry "%PYTHON_DIR%\python.exe" -m pip install --no-cache-dir virtualenv
if !errorlevel! neq 0 (
    echo Error: Failed to install virtualenv after %MAX_RETRIES% attempts.
    pause
    exit /b 1
)
echo Done.

:: Step 5: Create virtual environment and install dependencies
echo.
echo [5/6] Creating virtual environment and installing dependencies...
"%PYTHON_DIR%\python.exe" -m virtualenv --no-download "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo Error: Failed to create the virtual environment.
    pause
    exit /b 1
)
call "%VENV_DIR%\Scripts\activate"

echo Upgrading pip...
call :RunCommandWithRetry pip install --no-cache-dir --upgrade pip
if !errorlevel! neq 0 ( echo Error: Failed to upgrade pip. & pause & exit /b 1)

echo Installing main dependencies...
call :RunCommandWithRetry pip install --no-cache-dir transformers librosa gradio numpy scipy soundfile sentencepiece "huggingface_hub[hf_xet]" ffmpeg-python
if !errorlevel! neq 0 ( echo Error: Failed to install main dependencies. & pause & exit /b 1)

:: INSTALL PYTORCH WITH CUDA SUPPORT
echo.
echo Installing PyTorch with NVIDIA CUDA support...
call :RunCommandWithRetry pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if !errorlevel! neq 0 (
    echo Error: Failed to install PyTorch with CUDA support.
    pause
    exit /b 1
)
echo PyTorch for CUDA installed successfully.
echo Done.

:: Step 6: Create the launcher script
echo.
echo [6/6] Creating 'run_demo.bat' launcher...
(
echo @echo off
echo chcp 65001 ^> nul
echo setlocal enabledelayedexpansion
echo cd /d "%%~dp0"
echo set "PATH=%%~dp0PortableGit\bin;%%~dp0python_embedded;%%PATH%%"
echo set "TEMP=%%~dp0temp"
echo set "TMP=%%~dp0temp"
echo set "HF_HOME=%%~dp0huggingface_cache"
echo if not exist "%%TEMP%%" mkdir "%%TEMP%%"
echo if not exist "%%HF_HOME%%" mkdir "%%HF_HOME%%"
echo call "venv\Scripts\activate.bat"
echo echo.
echo echo Launching Borealis Gradio Demo...
echo echo Note: The first launch will download the model files, which may take some time.
echo echo.
echo python gradio_demo.py
echo pause
) > "%INSTALL_DIR%run_demo.bat"
echo Done.

echo.
echo -------------------------------------------------------------------------
echo Installation completed successfully!
echo -------------------------------------------------------------------------
echo.
echo To run Borealis Demo, use the 'run_demo.bat' file.
echo.
rmdir /s /q "%TEMP%" 2>nul
pause
goto :eof

:: ===========================================================================
:: Subroutines
:: ===========================================================================

:DownloadFile
set "URL=%~1"
set "OutputFile=%~2"
for /L %%i in (1,1,%MAX_RETRIES%) do (
    echo Attempt %%i of %MAX_RETRIES% to download %OutputFile%...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('!URL!', '!OutputFile!')"
    if !errorlevel! equ 0 (
        echo Download successful.
        exit /b 0
    )
    echo Download failed. Retrying in %RETRY_DELAY% seconds...
    timeout /t %RETRY_DELAY% /nobreak > nul
)
exit /b 1

:RunCommandWithRetry
set "command_to_run=%*"
for /L %%i in (1,1,%MAX_RETRIES%) do (
    echo Attempt %%i of %MAX_RETRIES%: Running '%command_to_run%'...
    %command_to_run%
    if !errorlevel! equ 0 (
        echo Command successful.
        exit /b 0
    )
    echo Command failed. Retrying in %RETRY_DELAY% seconds...
    timeout /t %RETRY_DELAY% /nobreak > nul
)
exit /b 1