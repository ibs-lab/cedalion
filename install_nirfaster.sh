#!/bin/bash
# Download and install NIRFASTer

echo "Installing NIRFASTer..."

OS="$(uname -s)"

case "$OS" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="mac";;
    CYGWIN*|MINGW*|MSYS*) OS_NAME="win";;
    *)          OS_NAME="Unknown";;
esac

echo "Operating System Detected: $OS_NAME"
if [ "$OS_NAME" = "Unknown" ]; then
    echo "Error: Failed at detecting the operating system. Plaese visit the NIRFASTer documentation and install it on your system: https://github.com/milabuob/nirfaster-uFF";
    exit 1
fi

if [ "$OS_NAME" = "mac" -a $1 = 'GPU' ]; then
    echo "Error: No releases are available for your configuration. please visit: https://github.com/milabuob/nirfaster-uFF";
    exit 1
fi

ZIP_URL="https://github.com/milabuob/nirfaster-uFF/archive/refs/heads/main.zip" 
ZIP_FILE="nirfaster-uFF-main.zip"


# wget -qO- "$ZIP_URL"| tar xvz -C .
wget -qO- "$ZIP_URL" > temp.zip && unzip temp.zip -d . && rm temp.zip

FOLDER_NAME=$(basename "$ZIP_FILE" .zip)
mv "${FOLDER_NAME}" "${FOLDER_NAME%-main}"


SOURCE_URL="https://github.com/milabuob/nirfaster-uFF/releases/download/v0.9.6/"

if [ $1 = 'CPU' ]; then
    # wget -qO- "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip"| tar xvz -C "${FOLDER_NAME%-main}""/nirfasteruff/" 
    wget -qO temp.zip "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip" && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip

elif [ $1 = 'GPU' ]; then
    # wget -qO- "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip"| tar xvz -C "${FOLDER_NAME%-main}""/nirfasteruff/";
    wget -qO temp.zip "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip" && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip

    # wget -qO- "$SOURCE_URL""gpu-"$OS_NAME"-python311.zip"| tar xvz -C "${FOLDER_NAME%-main}""/nirfasteruff/"
    wget -qO temp.zip "$SOURCE_URL""gpu-"$OS_NAME"-python311.zip" && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip

fi

if [ "$OS_NAME" = 'mac' ]; then
    xattr -c "${FOLDER_NAME%-main}/nirfasteruff/nirfasteruff_cpu.cpython-311-darwin.so"
fi

echo "NIRFASTer installed successfully."