#!/bin/bash

# 数据量
DATA_SIZE=$2

# 根据数据量决定进程数
if [ $DATA_SIZE -lt 2500 ]; then
    NUM_PROCESSES=1
elif [ $DATA_SIZE -lt 14000 ]; then
    NUM_PROCESSES=9
elif [ $DATA_SIZE -lt 21000 ]; then
    NUM_PROCESSES=14
elif [ $DATA_SIZE -lt 61000 ]; then
    NUM_PROCESSES=28
else
    NUM_PROCESSES=56
fi

# 检查是否分配到多机上
if [ $NUM_PROCESSES -gt 28 ]; then
  srun -N 2 -n $NUM_PROCESSES --cpu-bind=none ./wrapper.sh $*
else
  srun -N 1 -n $NUM_PROCESSES --cpu-bind=none ./wrapper.sh $*
fi
