@echo off
REM ElderCare web dashboard - one click run (double-click this file)
cd /d %~dp0
echo ============================================
echo  ElderCare dashboard is starting...
echo  Wait about 30 seconds on first run, then open:
echo  http://localhost:5000
echo  If the browser shows an error, wait a few
echo  seconds and press Refresh (F5).
echo  Keep this window OPEN while using the site.
echo ============================================
timeout /t 4 /nobreak >nul
start "" http://localhost:5000
python web.py
echo.
echo Server stopped. If you saw a red error above, send its text.
pause
