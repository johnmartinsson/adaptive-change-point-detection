wget https://zenodo.org/records/10811797/files/results_eusipco_2024.zip
wget https://zenodo.org/records/10811797/files/data.zip
unzip data.zip
unzip results_eusipco_2024.zip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# produces all tables in the paper
python src/tables.py

# produces all figures in the paper
python src/figures.py
