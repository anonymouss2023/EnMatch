#!/bin/bash
PATH="/project/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
echo "1"
source /project/.bash_profile
echo "1"
source /project/miniconda3/etc/profile.d/conda.sh
conda activate enmatch
export PYTHONPATH=/project/encom/enmatch
cd /project/encom/enmatch/script/strategy && export PYTHONIOENCODING=utf8 && python -u demo_elo.py &>> glomatch.out
echo "1"