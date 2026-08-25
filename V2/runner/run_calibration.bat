@echo off
rem Calibration pass - three regional batches in one go.
rem Night batches (cn, kr) come first. Putting us first would make --wait
rem sleep until 10:00 tomorrow and waste tonight's window.
rem
rem Run: double-click, or   run_calibration.bat
rem Safe to interrupt - the runner resumes from successful calls only.

chcp 65001 >nul
cd /d "%~dp0"
setlocal
set FAILED=

echo.
echo ================================================================
echo  [cn] DeepSeek, Qwen        window KST 23-07
echo ================================================================
call uv run calibrate.py --region cn --k 5 --wait
if errorlevel 1 set FAILED=%FAILED% cn

echo.
echo ================================================================
echo  [kr] Upstage, HyperCLOVA   window KST 23-07
echo ================================================================
call uv run calibrate.py --region kr --k 5 --wait
if errorlevel 1 set FAILED=%FAILED% kr

echo.
echo ================================================================
echo  [us] OpenAI, Google, Anthropic + anchors   window KST 10-17
echo ================================================================
call uv run calibrate.py --region us --k 5 --wait --min-remaining 2
if errorlevel 1 set FAILED=%FAILED% us

echo.
echo ================================================================
if defined FAILED goto :failed

echo  All three batches completed.
echo.
echo  Next:
echo    uv run select_bank.py     item bank 300 + noise floor
echo    uv run budget.py          projection + spend cap
goto :end

:failed
echo  FAILED:%FAILED%
echo.
echo  Re-run the failed region. The runner resumes from remaining calls:
echo    uv run calibrate.py --region ^<region^> --k 5 --wait
echo.
echo  All three must finish before select_bank.py.

:end
echo ================================================================
pause
