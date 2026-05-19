@echo off
cd /d %~dp0
call D:\softwaredownload\anaconda3\Scripts\activate.bat arcface_cpu
start "" pythonw run_app.py
exit