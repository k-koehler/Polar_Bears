#!/bin/bash
{
    git --version
} || {
    echo "Please install git"
    exit 1
}

{
    python3 --version 1>/dev/null
} || {
    echo "Please install python3"
    exit 1
}

{
    pip3 --version 1>/dev/null
} || {
    echo "Please install pip3"
    exit 1
}

cd ~
echo "Installing requirements..."
pip3 install -r requirements.txt
echo "Cloning repo..."
git clone https://github.com/k-koehler/Polar_Bears/
cd Polar_Bears/Implement
python3 predictor.py