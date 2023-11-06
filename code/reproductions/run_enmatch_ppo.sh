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


cd ${script_dir}

# experiment in a_all env, train in a_all sample and test in a_all sample
#python -u strategy/enmatch_v2.py "{'num_matches':1,'num_per_team':3,'sample_file':'${enmatch_dataset_dir}/simplematch_6_3_100_100000_linear_opt.csv'}" >> ${enmatch_output_dir}/enmatch_v2_6_3.log &&
python -u strategy/enmatch_v2.py "{'num_matches':3,'num_per_team':3,'sample_file':'${enmatch_dataset_dir}/simplematch_18_3_100_100000_linear_opt.csv'}" >> ${enmatch_output_dir}/enmatch_v2_18_3.log &&
echo "1"
