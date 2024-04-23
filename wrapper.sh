#!/bin/bash

# LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK # for OpenMPI
# LOCAL_RANK=$MPI_LOCALRANKID # for Intel MPI
LOCAL_RANK=$SLURM_LOCALID # for SLURM

# LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE # for OpenMPI
# LOCAL_SIZE=$MPI_LOCALNRANKS # for Intel MPI
LOCAL_SIZE=$SLURM_TASKS_PER_NODE # for SLURM

NCPUS=$(nproc --all) # eg: 28
NUM_NUMA=2

# calculate binding parameters
# bind to sequential cores in a NUMA domain
CORES_PER_NUMA=$(($NCPUS / $NUM_NUMA)) # 每一个NUMA节点上的核数
NUMA_ID=$(($LOCAL_RANK / $CORES_PER_NUMA)) # 进程的NUMA节点ID
NUMA_RANK=$(($LOCAL_RANK % $CORES_PER_NUMA)) # 进程在NUMA节点中的Rank
CORE=$(($NUM_NUMA * $NUMA_RANK + $NUMA_ID)) # 进程绑定的核

# execute command with specific cores
echo "Process $LOCAL_RANK on $(hostname) bound to core $CORE"
exec numactl -C "$CORE" $@