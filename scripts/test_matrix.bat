@echo off
REM Build + pytest matrix across Python versions and numpy/pandas/scipy version ranges.
REM Each combination gets an isolated venv via `uv venv`; uses `uv pip install` to
REM pin that combination's library versions, `uv build` to test packaging, and
REM pytest to run the test suite. Combinations that can't resolve/install are
REM reported as SKIP rather than FAIL (e.g. no wheels for that Python/lib pair).

set ROOT=%~dp0..
set VENV_DIR=%ROOT%\.venv-matrix
set DIST_DIR=%ROOT%\dist-matrix
set LOGFILE=%ROOT%\matrix_results.log

if exist "%LOGFILE%" del "%LOGFILE%"
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"

set PYTHONS=3.11 3.12 3.13 3.14
REM Exact pins (not ranges) are used throughout: cmd.exe treats "<" and ">" as
REM redirection operators even inside quotes, and "," splits FOR's (set) list
REM items even inside quotes, so range syntax like "numpy>=1.26,<2" is unsafe here.
set NUMPYS="numpy==1.26.4" "numpy==2.5.0"
set PANDASES="pandas==1.5.3" "pandas==2.2.3" "pandas==3.0.3"
set SCIPIES="scipy==1.11.4" "scipy==1.14.1" "scipy==1.18.0"
REM numba 0.59.1 = last 0.5x release; 0.60.0 = first 0.6x release; 0.65.1 = current stable
set NUMBAS="numba==0.59.1" "numba==0.60.0" "numba==0.65.1"

set /a TOTAL=0
set /a PASS=0
set /a FAIL=0
set /a SKIP=0

for %%P in (%PYTHONS%) do (
  for %%N in (%NUMPYS%) do (
    for %%A in (%PANDASES%) do (
      for %%S in (%SCIPIES%) do (
        for %%B in (%NUMBAS%) do (
          call :run_combo %%P %%N %%A %%S %%B
        )
      )
    )
  )
)

echo. >> "%LOGFILE%"
echo ===== SUMMARY ===== >> "%LOGFILE%"
echo Total: %TOTAL%  Passed: %PASS%  Failed: %FAIL%  Skipped: %SKIP% >> "%LOGFILE%"

echo.
echo ===== SUMMARY =====
echo Total: %TOTAL%  Passed: %PASS%  Failed: %FAIL%  Skipped: %SKIP%
echo Full log: %LOGFILE%

if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
exit /b %FAIL%

:run_combo
set PY=%~1
set NP=%~2
set PD=%~3
set SC=%~4
set NB=%~5
set /a TOTAL+=1

echo.
echo [%TOTAL%] python=%PY% %NP% %PD% %SC% %NB%
echo ----- [%TOTAL%] python=%PY% %NP% %PD% %SC% %NB% ----- >> "%LOGFILE%"

if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
uv venv --python %PY% "%VENV_DIR%" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo   -^> SKIP ^(python %PY% unavailable^)
  set /a SKIP+=1
  goto :eof
)

uv pip install --python "%VENV_DIR%\Scripts\python.exe" %NP% %PD% %SC% %NB% "holidays>=0.58" python-dateutil pytest >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo   -^> SKIP ^(dependency install failed^)
  set /a SKIP+=1
  goto :eof
)

uv build "%ROOT%" --out-dir "%DIST_DIR%" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo   -^> FAIL ^(build^)
  set /a FAIL+=1
  goto :eof
)

uv pip install --python "%VENV_DIR%\Scripts\python.exe" --no-deps -e "%ROOT%" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo   -^> FAIL ^(install package^)
  set /a FAIL+=1
  goto :eof
)

"%VENV_DIR%\Scripts\python.exe" -m pytest "%ROOT%\tests" -q >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo   -^> FAIL ^(pytest^)
  set /a FAIL+=1
  goto :eof
)

echo   -^> PASS
set /a PASS+=1
goto :eof
