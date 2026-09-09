@echo off
chcp 65001 >nul
cd /d "%~dp0"
uv run calibrate.py --region cn --k 5 --wait
uv run calibrate.py --region kr --k 5 --wait
uv run calibrate.py --region us --k 5 --wait --min-remaining 2