dataset_name=me
bg_label=all
fg_label=me
snr=10
n_soundscapes=50
data_dir=/mnt/storage_1/datasets/bioacoustic_sed

eval "$(conda shell.bash hook)"
conda activate torchaudio

python generate_scaper_data.py --dataset_name=$dataset_name --snr=$snr --bg_label=$bg_label --fg_label=$fg_label --n_soundscapes=$n_soundscapes --data_dir=$data_dir

conda activate birdnet
cd ~/gits/BirdNET-Analyzer/
embeddings_train_dir=$data_dir/generated_datasets/$fg_label/train_soundscapes_snr_$snr.0/ 
embeddings_test_dir=$data_dir/generated_datasets/$fg_label/test_soundscapes_snr_$snr.0/ 

python3 embeddings.py --i $embeddings_train_dir --o $embeddings_train_dir --overlap 2.5 --threads 8
python3 embeddings.py --i $embeddings_test_dir --o $embeddings_test_dir --overlap 2.5 --threads 8
