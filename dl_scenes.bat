@echo off

if not exist models (
    mkdir models
)

pushd models
git clone https://github.com/KhronosGroup/glTF-Sample-Models.git --depth 1
popd
