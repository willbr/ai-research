@echo off
setlocal

where zig >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Zig compiler not found in PATH!
    exit /b 1
)

set RAYLIB_INCLUDE=%USERPROFILE%\scoop\apps\vcpkg\current\installed\x64-windows\include
set RAYLIB_LIB=%USERPROFILE%\scoop\apps\vcpkg\current\installed\x64-windows\lib

if not exist "%RAYLIB_INCLUDE%" (
    echo Raylib include directory not found!
    exit /b 1
)
if not exist "%RAYLIB_LIB%" (
    echo Raylib lib directory not found!
    exit /b 1
)

zig cc main.c -o image.exe ^
  -I%RAYLIB_INCLUDE% ^
  -L%RAYLIB_LIB% ^
  -lraylib ^
  -lgdi32 -lwinmm -luser32 -lshell32 ^
  -target x86_64-windows ^
  -O2

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo Build succeeded! Run image.exe to play.
endlocal
