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

# Define the directory name
DIR_NAME="plugins"

# Check if the directory exists
if [ ! -d "$DIR_NAME" ]; then
  # If the directory does not exist, create it
  mkdir -p "$DIR_NAME"
fi

cd "$DIR_NAME"

ZIP_URL="https://codeload.github.com/milabuob/nirfaster-uFF/zip/refs/tags/v1.0.0" 
ZIP_FILE="nirfaster-uFF-main.zip"


curl -sL "$ZIP_URL" -o temp.zip && unzip temp.zip -d . && rm temp.zip

FOLDER_NAME=$(basename "$ZIP_FILE" .zip)
mv "${FOLDER_NAME}" "${FOLDER_NAME%-main}"


SOURCE_URL="https://github.com/milabuob/nirfaster-uFF/releases/download/v1.0.0/"

if [ $1 = 'CPU' ]; then
    curl -sL "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip" -o temp.zip && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip

elif [ $1 = 'GPU' ]; then
    curl -sL "$SOURCE_URL""cpu-"$OS_NAME"-python311.zip" -o temp.zip && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip
    curl -sL "$SOURCE_URL""gpu-"$OS_NAME"-python311.zip" -o temp.zip && unzip temp.zip -d "${FOLDER_NAME%-main}/nirfasteruff/" && rm temp.zip


fi

if [ "$OS_NAME" = 'mac' ]; then
    xattr -c "${FOLDER_NAME%-main}/nirfasteruff/nirfasteruff_cpu.cpython-311-darwin.so"
fi

if [ "$OS_NAME" = 'linux' ]; then
    chmod +x "${FOLDER_NAME%-main}/nirfasteruff/cgalmesherLINUX"
fi

echo "NIRFASTer installed successfully."