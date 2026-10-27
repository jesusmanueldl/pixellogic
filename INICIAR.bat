@echo off
cd /d "%~dp0"
py -3 studio.py
if errorlevel 1 pause
