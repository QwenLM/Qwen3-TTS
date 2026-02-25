@echo off
REM Master script to run all Phase 1 diagnostics in sequence
REM Usage: run_phase1_diagnostics.bat

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo QWEN3 TTS - PHASE 1 DIAGNOSTICS MASTER SCRIPT
echo ======================================================================
echo.
echo This will run 5 diagnostic scripts to identify the root cause
echo of the transformers 5.x silence bug.
echo.
echo Transformers version:
python -c "import transformers; print(f'  {transformers.__version__}')"
echo.
echo ======================================================================
echo.

REM Create simple timestamp for results directory
for /f "tokens=*" %%a in ('python -c "from datetime import datetime; print(datetime.now().strftime('%%Y%%m%%d_%%H%%M%%S'))"') do set TIMESTAMP=%%a
set RESULTS_DIR=diag_results_%TIMESTAMP%
mkdir "%RESULTS_DIR%" 2>nul

echo Results will be saved to: %RESULTS_DIR%
echo.

REM Step 1: Environment Check
echo ======================================================================
echo STEP 1/5: Environment Check (no GPU required)
echo ======================================================================
python diag_env_check.py > "%RESULTS_DIR%\01_env_check.log" 2>&1
if !errorlevel! equ 0 (
    echo [OK] Environment check complete
) else (
    echo [ERROR] Environment check failed - see 01_env_check.log
)
echo.

REM Step 2: Static Analysis
echo ======================================================================
echo STEP 2/5: Static Analysis (no GPU required)
echo ======================================================================
python diag_static_analysis.py > "%RESULTS_DIR%\02_static_analysis.log" 2>&1
if !errorlevel! equ 0 (
    echo [OK] Static analysis complete
) else (
    echo [ERROR] Static analysis failed - see 02_static_analysis.log
)
echo.

REM Step 3: RoPE Tensors
echo ======================================================================
echo STEP 3/5: RoPE Tensor Comparison (CPU ok)
echo ======================================================================
python diag_rope_tensors.py > "%RESULTS_DIR%\03_rope_tensors.log" 2>&1
if !errorlevel! equ 0 (
    echo [OK] RoPE tensor extraction complete
    if exist rope_inv_freq_computed.pt (
        echo [COPY] Copying RoPE tensor files...
        for %%F in (rope_*.pt) do (
            copy "%%F" "%RESULTS_DIR%\" >nul 2>&1
        )
    )
) else (
    echo [ERROR] RoPE tensor extraction failed - see 03_rope_tensors.log
)
echo.

REM Step 4: Mask Capture
echo ======================================================================
echo STEP 4/5: Causal Mask Capture (requires GPU)
echo ======================================================================
python diag_causal_mask_capture.py > "%RESULTS_DIR%\04_mask_capture.log" 2>&1
if !errorlevel! equ 0 (
    echo [OK] Mask capture complete
    if exist mask_captures.json (
        copy "mask_captures.json" "%RESULTS_DIR%\" >nul 2>&1
    )
) else (
    echo [ERROR] Mask capture failed - see 04_mask_capture.log
)
echo.

REM Step 5: Position IDs Trace
echo ======================================================================
echo STEP 5/5: Position IDs and cache_position Trace (requires GPU)
echo ======================================================================
python diag_position_ids.py > "%RESULTS_DIR%\05_position_ids.log" 2>&1
if !errorlevel! equ 0 (
    echo [OK] Position trace complete
    if exist position_trace.json (
        copy "position_trace.json" "%RESULTS_DIR%\" >nul 2>&1
    )
) else (
    echo [ERROR] Position trace failed - see 05_position_ids.log
)
echo.

echo ======================================================================
echo PHASE 1 DIAGNOSTICS COMPLETE
echo ======================================================================
echo.
echo Results saved to: %RESULTS_DIR%
echo.
echo Log files:
echo   - %RESULTS_DIR%\01_env_check.log
echo   - %RESULTS_DIR%\02_static_analysis.log
echo   - %RESULTS_DIR%\03_rope_tensors.log
echo   - %RESULTS_DIR%\04_mask_capture.log
echo   - %RESULTS_DIR%\05_position_ids.log
echo.
echo Data files:
echo   - %RESULTS_DIR%\rope_*.pt (tensor files)
echo   - %RESULTS_DIR%\mask_captures.json
echo   - %RESULTS_DIR%\position_trace.json
echo.
echo Next steps:
echo 1. Check the log files for any errors
echo 2. Archive baseline (if transformers 4.x):
echo    mkdir rope_4x_backup
echo    copy "%RESULTS_DIR%\rope_*.pt" rope_4x_backup\
echo    copy "%RESULTS_DIR%\mask_captures.json" mask_captures_4x.json
echo    copy "%RESULTS_DIR%\position_trace.json" position_trace_4x.json
echo.
echo 3. Switch to transformers 5.x, run this script again
echo.
echo 4. Compare outputs using the decision tree in DIAGNOSTIC_PLAN.md
echo.
echo See DIAGNOSTIC_QUICK_REFERENCE.md for detailed analysis instructions.
echo.

pause
