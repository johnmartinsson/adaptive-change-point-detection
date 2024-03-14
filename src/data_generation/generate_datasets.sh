base_dir=/mnt/storage_1/john/data/bioacoustic_sed_2024_02_22/
# produce all the source material for the bioacoustic_sed project
# generate the synthetic datasets
n_soundscapes=300
# NOTE: these can be run in parallell to reduce time
# TODO: automate, or explain BirdNET-Analyzer installationn

#echo "preparing sound source material ..."
#python produce_source_material.py --source_base_dir=$source_base_dir

echo "generating synthetic datasets for all classes ..."
for class_name in me dog baby
do
    bash scripts/generate_scaper_data.sh $class_name $base_dir $n_soundscapes
done