#!/bin/bash

script_abs=$(readlink -f "$0")
enmatch_benchmark_dir=$(dirname $script_abs)/..
enmatch_dataset_dir=${enmatch_benchmark_dir}/dataset
script_dir=${enmatch_benchmark_dir}/script
enmatch_output_dir=${enmatch_benchmark_dir}/output
export enmatch_benchmark_dir && export enmatch_output_dir && export enmatch_dataset_dir

file=$1

cd ${enmatch_dataset_dir} &&

awk -F "@" '$2%11<2 {print}' ${file} > ${enmatch_output_dir}/${file}_0000.csv &&
awk -F "@" '$2%11>=2 && $2%11<4 {print}' ${file} > ${enmatch_output_dir}/${file}_0001.csv &&
awk -F "@" '$2%11>=4 && $2%11<6 {print}' ${file} > ${enmatch_output_dir}/${file}_0002.csv &&
awk -F "@" '$2%11>=6 && $2%11<8 {print}' ${file} > ${enmatch_output_dir}/${file}_0003.csv &&
awk -F "@" '$2%11>=8 {print}' ${file} > ${enmatch_output_dir}/${file}_0004.csv

#file_rows=`wc -l ${file}|awk '{print $1}'`
#file_num=5
#file_num_row=$((${file_rows} + 4))
#every_file_row=$((${file_num_row}/${file_num}))
#split -d -a 4 -l ${every_file_row} ${file} --additional-suffix=.csv ${enmatch_output_dir}/${file}_


