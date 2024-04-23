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
  // 准备数据
  float *sort_arr = new float[((n + nprocs - 1) / nprocs) * 2]; // 排序数组
  memcpy(sort_arr, data, block_len * sizeof(float));            // 将数据写入排序数组
  std::sort(sort_arr, sort_arr + block_len);                    // 第一次排序

  // 交换数据
  bool no_swap[2] = {0, 0};
  for (bool stage = 0;; stage ^= 1) // stage 0: 偶数阶段，stage 1: 奇数阶段
  {
    if (rank % 2 == stage) // 接收进程（排序进程）
    {
      if (rank != nprocs - 1) // 不在最后一个进程：接收右侧数据
      {
        float my_max = sort_arr[block_len - 1];
        float right_min;
        MPI_Sendrecv(&my_max, 1, MPI_FLOAT, rank + 1, 0,
                     &right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_max > right_min) // 需要交换
        {
          float *beg_pos = std::lower_bound(sort_arr, sort_arr + block_len, right_min); // 查找排序开始的位置
          int recv_len;
          MPI_Recv(&recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                     // 接收待排序数据长度
          MPI_Recv(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // 接收待排序数据
          std::inplace_merge(beg_pos, &sort_arr[block_len], &sort_arr[block_len + recv_len]);                  // 合并数据（排序）
          MPI_Send(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);                    // 发送多出的数据
          no_swap[stage] = false;
        }
        else // 不需要交换
          no_swap[stage] = true;
      }
      else // 最后一个进程：不需要交换
        no_swap[stage] = true;
    }
    else // 发送进程（提供数据的进程）
    {
      if (rank != 0) // 不在第一个进程：发送左侧数据
      {
        float my_min = sort_arr[0];
        float left_max;
        MPI_Sendrecv(&my_min, 1, MPI_FLOAT, rank - 1, 0,
                     &left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max) // 需要交换
        {
          float *end_pos = std::upper_bound(sort_arr, sort_arr + block_len, left_max); // 查找排序结束的位置
          int recv_len = end_pos - sort_arr;
          MPI_Send(&recv_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);                            // 发送待排序数据长度
          MPI_Send(sort_arr, recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);                    // 发送待排序数据
          MPI_Recv(sort_arr, recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // 接收新数据
          no_swap[stage] = false;
        }
        else // 不需要交换
          no_swap[stage] = true;
      }
      else // 第一个进程：不需要交换
        no_swap[stage] = true;
    }
    if (stage) // 奇数阶段，检查
    {
      bool my_flag = no_swap[0] && no_swap[1], left_flag, right_flag;
      for (int i = 1; i < nprocs; ++i) // 进程间相互通知
      {
        if (rank == 0)
        {
          MPI_Send(&my_flag, 1, MPI_C_BOOL, 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&left_flag, 1, MPI_C_BOOL, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          my_flag &= left_flag;
        }
        else if (rank == nprocs - 1)
        {
          MPI_Recv(&right_flag, 1, MPI_C_BOOL, nprocs - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(&my_flag, 1, MPI_C_BOOL, nprocs - 2, 0, MPI_COMM_WORLD);
          my_flag &= right_flag;
        }
        else
        {
          MPI_Sendrecv(&my_flag, 1, MPI_C_BOOL, rank + 1, 0,
                       &left_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Sendrecv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0,
                       &right_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          my_flag &= left_flag & right_flag;
        }
      }
      if (my_flag)
        break;
    }
  }

  // 写回数据
  memcpy(data, sort_arr, block_len * sizeof(float));
}
