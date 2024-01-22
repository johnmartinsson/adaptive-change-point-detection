# TODO: download the NIGENS, TUT and DCASE datasets

source_base_dir=/mnt/storage_1/john/data/bioacoustic_sed/sources/
base_dir=/mnt/storage_1/john/data/bioacoustic_sed/
# produce all the source material for the bioacoustic_sed project

echo "preparing sound source material ..."
#python produce_source_material.py --source_base_dir=$source_base_dir

# generate the synthetic datasets
n_soundscapes=300
# NOTE: these can be run in parallell to reduce time
# TODO: automate, or explain BirdNET-Analyzer installationn

echo "generating synthetic datasets for all classes ..."
# for class_name in me dog baby
# do
#     bash scripts/generate_scaper_data.sh $class_name $base_dir $n_soundscapes
# done

# run the simulations
# NOTE: these can be run in paralell to reduce time

n_runs=1
results_dir=$1/${n_runs}_runs/
n_queries_budget=$3
noise=$2
only_budget_1=$4

# create results dir if not exists
mkdir -p ${results_dir}

# echo start time to file
#settings="noise_${noise}_n_queries_${n_queries_budget}"
echo "start time: $(date)" >> ${results_dir}/time.txt

#########################
# Simulations
#########################

# noisy oracle with accuracy 1.0 - noise
echo "running annotation simulations for all classes ..."
echo "noise: $noise"
for class_name in me dog baby
do
    python run_simulation.py --base_dir=$base_dir --results_dir=${results_dir}/ --class_name=$class_name --n_soundscapes_budget=$n_soundscapes --n_queries_budget=$n_queries_budget --n_runs=$n_runs --emb_win_length=1.0 --fp_noise=$noise --fn_noise=$noise
done

#########################
# Prediction
#########################

# compute training and test prediction scores
# echo "computing training and test prediction scores for all classes and evaluation models ..."
# for class_name in me dog baby
# do
#     for model_name in mlp prototypical
#     do        
#         bash ./scripts/evaluate_all.sh $class_name ${results_dir}/$class_name/n_queries_${n_queries_budget}_noise_${noise}/ $model_name $base_dir $n_runs $only_budget_1
#     done
# done

#########################
# Evaluation
#########################

t_collar=2.0
# evaluate the results
# echo "evaluating the results for all classes and evaluation models ..."
# for class_name in me dog baby
# do
#     for model_name in mlp prototypical
#     do        
#         python sound_event_eval.py --class_name=$class_name --model_name=$model_name --t_collar=$t_collar --n_runs=$n_runs --sim_dir=${results_dir}/$class_name/n_queries_${n_queries_budget}_noise_${noise}/ --base_dir=$base_dir --only_budget_1=$only_budget_1
#     done
# done

# echo end time to file
echo "end time: $(date)" >> ${results_dir}/time.txt