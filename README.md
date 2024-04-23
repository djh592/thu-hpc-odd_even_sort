# 实验一：奇偶排序（odd_even_sort）

2024 春高性能计算导论 PA1

## 代码

```cpp
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort()
{
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  // 防止 nprocs > n
  if (out_of_range)
    return;

  // 单进程避免读写
  if (nprocs == 1)
  {
    std::sort(data, data + n);
    return;
  }

  // 准备常量
  const int parity = rank % 2;                                  // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;             // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size;      // 参与排序的进程数
  const int l_neighbour = rank > 0 ? rank - 1 : -1;             // 左侧邻居
  const int r_neighbour = rank < proc_size - 1 ? rank + 1 : -1; // 右侧邻居
  const int max_swap = 2 * (proc_size - 1);                     // 最大交换次数

  // 数据量小，减少通信次数的算法
  if (n < 20000000)
  {
    // 写入数据
    float *sort_arr = new float[block_size * 3]; // 排序数组
    float *my_arr_beg = &sort_arr[block_size];
    float *my_arr_end = my_arr_beg + block_len;
    float *l_recv_arr = new float[block_size];
    memcpy(my_arr_beg, data, block_len * sizeof(float)); // 将数据写入排序数组
    std::sort(my_arr_beg, my_arr_end);                   // 第一次排序

    // 交换数据
    int send_rank = 0;
    MPI_Request *send_list = new MPI_Request[max_swap * 2]; // 统一 wait Isend
    // MPI_Request recv_req;                                   // 单独 wait Irecv
    MPI_Status recv_status; // 单独回收 Irecv
    for (int swap = 0, stage = 0; swap < max_swap; stage = (++swap) % 2)
    {
      if (parity == stage) // 左边进程（右侧交换）
      {
        if (r_neighbour != -1)
        {
          float my_max = *(my_arr_end - 1);
          float right_min;
          MPI_Isend(&my_max, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&right_min, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_max > right_min) // 需要交换
          {
            float *beg_pos = std::upper_bound(my_arr_beg, my_arr_end, right_min);
            int send_len = my_arr_end - beg_pos;
            MPI_Sendrecv(beg_pos, send_len, MPI_FLOAT, r_neighbour, 0,
                         my_arr_end, block_size, MPI_FLOAT, r_neighbour, 0,
                         MPI_COMM_WORLD, &recv_status);
            int recv_len;
            MPI_Get_count(&recv_status, MPI_FLOAT, &recv_len);
            std::inplace_merge(beg_pos, my_arr_end, my_arr_end + recv_len);
          }
        }
      }
      else // 右边进程（左侧交换）
      {
        if (l_neighbour != -1)
        {
          float my_min = *my_arr_beg;
          float left_max;
          MPI_Isend(&my_min, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&left_max, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_min < left_max) // 需要交换
          {
            float *end_pos = std::lower_bound(my_arr_beg, my_arr_end, left_max);
            int send_len = end_pos - my_arr_beg;
            MPI_Sendrecv(my_arr_beg, send_len, MPI_FLOAT, l_neighbour, 0,
                         l_recv_arr, block_size, MPI_FLOAT, l_neighbour, 0,
                         MPI_COMM_WORLD, &recv_status);
            int recv_len;
            MPI_Get_count(&recv_status, MPI_FLOAT, &recv_len);
            std::merge(my_arr_beg, end_pos, l_recv_arr, l_recv_arr + recv_len, &sort_arr[block_size - recv_len]);
          }
        }
      }
    }

    // 写回数据
    memcpy(data, my_arr_beg, block_len * sizeof(float));
    MPI_Waitall(send_rank, send_list, MPI_STATUSES_IGNORE);
    delete[] sort_arr;
    delete[] l_recv_arr;
    delete[] send_list;
  }
  // n 很大，减少通信传输量的算法
  else
  {
    // 写入数据
    float *sort_arr = new float[block_size * 2];       // 排序数组
    memcpy(sort_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
    std::sort(sort_arr, sort_arr + block_len);         // 第一次排序

    // 交换数据
    int req_rank = 0;
    MPI_Request *req_list = new MPI_Request[2 * max_swap]; // send_list：用于统一回收 Isend
    for (int swap = 0, stage = 0; swap < max_swap; stage = (++swap) % 2)
    {
      if (parity == stage) // 接收进程（右侧交换）
      {
        if (r_neighbour != -1)
        {
          float my_max = sort_arr[block_len - 1];
          float right_min;
          MPI_Isend(&my_max, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &req_list[req_rank++]);
          MPI_Recv(&right_min, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_max > right_min) // 需要交换
          {
            MPI_Irecv(&sort_arr[block_len], block_size, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &req_list[req_rank]);
            int recv_len;
            MPI_Status status;
            float *beg_pos = std::upper_bound(sort_arr, sort_arr + block_len, right_min);
            MPI_Wait(&req_list[req_rank], &status);
            MPI_Get_count(&status, MPI_FLOAT, &recv_len);
            std::inplace_merge(beg_pos, &sort_arr[block_len], &sort_arr[block_len + recv_len]);
            MPI_Isend(&sort_arr[block_len], recv_len, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &req_list[req_rank++]);
          }
        }
      }
      else // 发送进程（左侧交换）
      {
        if (l_neighbour != -1)
        {
          float my_min = sort_arr[0];
          float left_max;
          MPI_Isend(&my_min, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &req_list[req_rank++]);
          MPI_Recv(&left_max, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_min < left_max) // 需要交换
          {
            float *end_pos = std::lower_bound(sort_arr, sort_arr + block_len, left_max);
            int send_len = end_pos - sort_arr;
            MPI_Isend(sort_arr, send_len, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &req_list[req_rank++]);
            MPI_Recv(sort_arr, send_len, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
      }
    }

    // 写回数据
    memcpy(data, sort_arr, block_len * sizeof(float));
    MPI_Waitall(req_rank, req_list, MPI_STATUSES_IGNORE);
    delete[] sort_arr;
    delete[] req_list;
  }

  return;
}
```

**说明：**

首先，准备一个用来排序的数据 `sort_arr`，将数据写入，并用 `std::sort` 进行第一次排序。

开始交换，我写了两种交换策略：

- 相邻进程互换数据，然后 merge；这种方法有利于减少通信次数。

- rank 小的进程接收原始数据，并进行 `std::inplace_merge`，发送合并好的数据；rank 大的进程发送原始数据，接收合并好的数据；这种方法有利于减少传输数据量。

第一种方法在数据量较小时使用；第二种方法在数据量较大时使用。

交换完成后，将 `sort_arr` 中数据写回 `data`。

## 性能优化

在我的代码中，做出的优化如下：

- 数据量小时减少通信次数；数据量大时减少通信数据量

- 相邻进程交换数据前先检查是否要交换，不交换数据则跳过通信
- 使用 `std::upper_bound` 和 `std::lower_bound` 确定最少的交换数量
- 重排使用 `std::inplace_merge` 而非 `std::sort`，降低复杂度
- send 操作的 wait 时机不好确认，不做阻塞，最后统一 waitall；recv 操作的 wait 时机很明确，对所有的 recv 进行阻塞，由于 send 和 recv 的一一对应关系可以在交换过程中为 send 形成最迟的回收时机

- 不并行时不写入和写回 `data`，直接 `std::sort`

另外，我将进程和核绑定，相邻进程尽量在同一个 NUMA domain 中，见 `wrapper.sh`：

```shell
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
exec numactl -C "$CORE" $@
```

启动的进程数量和数据量有关，见 `run.sh`：

```shell
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
```

## 测试

使用 Makefile 编译代码，编译的选项为：`-std=c++14 -O3 -Wall -Wextra -Werror -Wno-cast-function-type -pedantic -funroll-loops -DDEBUG`。

使用 `srun -N <机器数> -n <进程数> --cpu-bind=none ./wrapper.sh $*` 运行代码，进行进程映射。

$1×1$，$1×2$，$1×4$，$1×8$，$1×16$，$2×16$ 进程及元素数量 $n=100000000$ 的情况下 `sort` 函数的运行时间，及相对单进程的加速比如下：

| 进程数 | 运行时间（ms） | 加速比   |
| ------ | -------------- | -------- |
| $1×1$  | 12411.246000   | 1        |
| $1×2$  | 6872.504000    | 1.805928 |
| $1×4$  | 3901.245000    | 3.181355 |
| $1×8$  | 2373.050000    | 5.230082 |
| $1×16$ | 1497.117000    | 8.290098 |
| $2×16$ | 949.361000     | 13.07326 |

注：上述测试的结果都是三次取平均得到。

