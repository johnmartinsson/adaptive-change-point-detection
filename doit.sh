wget <zenodo>/data.zip
wget <zenodo>/results_eusipco_2024.zip
unzip data.zip
unzip results_eusipco_2024.zip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
