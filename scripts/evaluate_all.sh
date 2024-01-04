python evaluate.py --sim_dir=$2 --emb_win_length=1.0 --strategy_name=OPT --class_name=$1 --n_runs=10 --model_name=$3
python evaluate.py --sim_dir=$2 --emb_win_length=1.0 --strategy_name=ADP --class_name=$1 --n_runs=10 --model_name=$3
python evaluate.py --sim_dir=$2 --emb_win_length=1.0 --strategy_name=FIX --class_name=$1 --n_runs=10 --model_name=$3
python evaluate.py --sim_dir=$2 --emb_win_length=1.0 --strategy_name=CPD --class_name=$1 --n_runs=10 --model_name=$3
