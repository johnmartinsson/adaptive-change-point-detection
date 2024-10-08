fg_label=$1
dataset_name=${fg_label}_1.0_0.25s
bg_label=all
snr=0
n_soundscapes=$3
data_dir=$2 #/mnt/storage_1/datasets/bioacoustic_sed

eval "$(conda shell.bash hook)"
conda activate torchaudio

python generate_scaper_data.py --dataset_name=$dataset_name --snr=$snr --bg_label=$bg_label --fg_label=$fg_label --n_soundscapes=$n_soundscapes --data_dir=$data_dir

conda activate birdnet
cd ~/gits/BirdNET-Analyzer/
embeddings_train_dir=$data_dir/generated_datasets/$dataset_name/train_soundscapes_snr_$snr.0/ 
embeddings_test_dir=$data_dir/generated_datasets/$dataset_name/test_soundscapes_snr_$snr.0/ 

python3 embeddings.py --i $embeddings_train_dir --o $embeddings_train_dir --overlap 2.75 --threads 8 --batchsize 16
python3 embeddings.py --i $embeddings_test_dir --o $embeddings_test_dir --overlap 2.75 --threads 8 --batchsize 16
