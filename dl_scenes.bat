@echo off

if exist models rmdir /s /q models
mkdir models
pushd models
git clone https://github.com/KhronosGroup/glTF-Sample-Models.git --depth 1
if exist glTF-Sample-Models\1.0 rmdir /s /q glTF-Sample-Models\1.0
popd
