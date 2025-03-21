#! /bin/bash
###
 # @Author: yooki(yooki.k613@gmail.com)
 # @LastEditTime: 2025-03-21 14:47:47
 # @Description: run main.py
### 
# for example: bash run.sh -h"
# for example: bash run.sh 1 "-cs 0,1,2,3 -c cpu_util,mem_util -gle 500 -w pdtw -rf -sh -lrs adaptive -n pdtw,full_workflow"
# for example: bash run.sh 3 "-cs 0,1,2,3 -c cpu_util,mem_util -gle 500 -w pdtw -gip -gnt -id 1722795773 -lrs adaptive -n pdtw,load_gan_5773"
if [ "$1" = "-h" ]; then
    echo "Usage: bash run.sh [run_times] [args]"
    python main.py -h
else
    for (( i=1; i<=$1; i++ ))
    do
        echo "Running iteration $i"
        python main.py $2 
    done
fi