#!/bin/bash
{
    git --version 1>/dev/null
} || {
    echo "Please install git"
    exit 1
}

echo "git requirement satisfied"

{
    python3 --version 1>/dev/null
} || {
    echo "Please install python3"
    exit 1
}

echo "python3 requirement satisfied"

{
    pip3 --version 1>/dev/null
} || {
    echo "Please install pip3"
    exit 1
}

echo "pip3 requirement satisfied"

read -p "Install in ~/Polar_Bears? [Y/n]" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]];
then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
    echo "Cloning repo..."
    cd ~
    git clone https://github.com/k-koehler/Polar_Bears/
    cd Polar_Bears/Implement
    python3 predictor.py
else
    echo "Okay, exiting."
fi