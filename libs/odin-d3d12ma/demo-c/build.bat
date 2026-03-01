@echo off
setlocal
set SDK_DIR=C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0
cl /TC /nologo /I"%SDK_DIR%\shared" /I"%SDK_DIR%\um" /I"..\d3d12ma" /Fe:demo-c.exe main.c /link /LIBPATH:".." d3d12.lib dxgi.lib dxguid.lib
