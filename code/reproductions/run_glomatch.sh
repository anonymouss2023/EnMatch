#!/bin/bash
PATH="/project/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
source /project/.bash_profile
source /project/miniconda3/etc/profile.d/conda.sh
conda activate enmatch

script_abs=$(readlink -f "$0")
enmatch_benchmark_dir=$(dirname $script_abs)/..
enmatch_output_dir=${enmatch_benchmark_dir}/output
enmatch_dataset_dir=${enmatch_benchmark_dir}/dataset
script_dir=${enmatch_benchmark_dir}/script
export enmatch_benchmark_dir && export enmatch_output_dir && export enmatch_dataset_dir
export PYTHONPATH=/project/encom/enmatch
export PYTHONIOENCODING=utf8

algo=$1

cd ${enmatch_benchmark_dir}/enmatch/server
nohup python -u gymHttpServer.py &>> nohup.out &


cd ${script_dir}

# experiment in a_all env, train in a_all sample and test in a_all sample
#python -u strategy/glomatch.py $algo "train" "{'gpu':1,'env':'SeqSimpleMatchRecEnv-v0','sample_file':'${enmatch_dataset_dir}/simplematch_6_3_100_100000_linear_opt.csv','max_steps':6,'num_players':6,'num_per_team':3,'num_matches':1,'trial_name':'_6_3','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/glomatch_6_3_${algo}.log &&
#python -u modelfree_train.py $algo "eval" "{'env':'SlateRecEnv-v0','iteminfo_file':'${enmatch_dataset_dir}/item_info.csv','sample_file':'${enmatch_dataset_dir}/enmatch_dataset_a_shuf.csv','model_file':'${enmatch_output_dir}/simulator_a_dien/model','trial_name':'a_all','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/modelfree_a_all_${algo}.log &&
#python -u modelfree_train.py $algo "ope" "{'env':'SlateRecEnv-v0','iteminfo_file':'${enmatch_dataset_dir}/item_info.csv','sample_file':'${enmatch_dataset_dir}/enmatch_dataset_a_shuf.csv','model_file':'${enmatch_output_dir}/simulator_a_dien/model','trial_name':'a_all','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/modelfree_a_all_${algo}.log &&


#python -u strategy/glomatch.py $algo "train" "{'gpu':1,'env':'SeqSimpleMatchRecEnv-v0','sample_file':'${enmatch_dataset_dir}/simplematch_18_3_100_100000_linear_opt.csv','max_steps':18,'num_players':18,'action_size':18,'num_per_team':3,'num_matches':3,'trial_name':'_18_3','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/glomatch_18_3_${algo}.log &&
python -u strategy/glomatch.py $algo "train" "{'gpu':1,'env':'SeqSimpleMatchRecEnv-v0','sample_file':'${enmatch_dataset_dir}/simplematch_20_10_1000_100000_linear_opt.csv','max_steps':20,'num_players':20,'action_size':20,'num_per_team':10,'num_matches':1,'trial_name':'_20_10','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/glomatch_20_10_${algo}.log &&
#python -u strategy/glomatch.py $algo "train" "{'gpu':1,'env':'SeqSimpleMatchRecEnv-v0','sample_file':'${enmatch_dataset_dir}/simplematch_60_10_1000_100000_linear_opt.csv','max_steps':60,'num_players':60,'action_size':60,'num_per_team':10,'num_matches':3,'trial_name':'_60_10','remote_base':'http://127.0.0.1:5000'}" >> ${enmatch_output_dir}/glomatch_60_10_${algo}.log &&

echo "1"
