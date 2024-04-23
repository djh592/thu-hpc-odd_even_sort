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

  if (nprocs == 0)
  {
    std::sort(data, data + n);
    return;
  }

  // 准备常量
  const int parity = rank % 2;                             // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;        // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size; // 参与排序的进程数
  const size_t l_block_len = rank ? block_size : 0;        // 左侧数据长度
  const size_t r_block_len = rank < proc_size - 2 ? block_size : rank == proc_size - 2 ? n - (proc_size - 1) * block_size
                                                                                       : 0; // 右侧数据长度

  if (block_size < 8000 || block_size > 15000) // 大数据的情况
  {
    // 写入数据
    float *sort_arr = new float[block_size * 2];       // 排序数组
    memcpy(sort_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
    std::sort(sort_arr, sort_arr + block_len);         // 第一次排序

    // 交换数据
    const int max_swap = 2 * (proc_size - 1); // 最大交换次数
    int send_rank = 0;
    MPI_Request *send_list = new MPI_Request[3 * max_swap]; // send_list：用于统一回收 Isend
    for (int swap = 0, stage = 0; swap < max_swap; stage = (++swap) % 2)
    {
      if (parity == stage) // 接收进程（右侧交换）
      {
        if (r_block_len)
        {
          float my_max = sort_arr[block_len - 1];
          float right_min;
          MPI_Isend(&my_max, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_max > right_min) // 需要交换
          {
            float *beg_pos = std::lower_bound(sort_arr, sort_arr + block_len, right_min);
            int recv_len;
            MPI_Recv(&recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::inplace_merge(beg_pos, &sort_arr[block_len], &sort_arr[block_len + recv_len]);
            MPI_Isend(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          }
        }
      }
      else // 发送进程（左侧交换）
      {
        if (l_block_len)
        {
          float my_min = sort_arr[0];
          float left_max;
          MPI_Isend(&my_min, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_min < left_max) // 需要交换
          {
            float *end_pos = std::upper_bound(sort_arr, sort_arr + block_len, left_max);
            int send_len = end_pos - sort_arr;
            MPI_Isend(&send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
            MPI_Isend(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
            MPI_Recv(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
      }
    }
    // 写回数据
    memcpy(data, sort_arr, block_len * sizeof(float));
    MPI_Waitall(send_rank, send_list, MPI_STATUSES_IGNORE);
    delete[] sort_arr;
  }
  else // 小数据的情况
  {
    float *sort_arr = new float[l_block_len + block_len + r_block_len]; // 排序数组
    float *l_arr = sort_arr;
    float *my_arr = l_arr + l_block_len;
    float *r_arr = my_arr + block_len;
    memcpy(my_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
    std::sort(my_arr, my_arr + block_len);           // 第一次排序
    // 小数据中，一个 swap 计入两次交换
    const int max_swap = (proc_size - 1) * 2;               // 最大交换次数
    MPI_Request *send_list = new MPI_Request[max_swap * 2]; // send_list：用于统一回收 Isend
    int send_rank = 0;
    MPI_Request requests[2]; // 用于异步通信(一轮最多 2 个 Wait)
    for (int swap = 0; swap < max_swap; swap++)
    {
      if (!l_block_len) // 第一个进程
      {
        MPI_Isend(my_arr, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(r_arr, r_block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::inplace_merge(my_arr, r_arr, r_arr + r_block_len);
      }
      else if (!r_block_len) // 最后一个进程
      {
        MPI_Isend(my_arr, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(l_arr, l_block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::inplace_merge(l_arr, my_arr, my_arr + block_len);
      }
      else // 中间进程
      {
        if (!parity) // 偶数进程
        {
          MPI_Irecv(l_arr, l_block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Irecv(r_arr, r_block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
          MPI_Isend(my_arr, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
          std::inplace_merge(my_arr, r_arr, r_arr + r_block_len);
          MPI_Isend(my_arr, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
          std::inplace_merge(l_arr, my_arr, my_arr + block_len);
        }
        else // 奇数进程
        {
          MPI_Irecv(l_arr, l_block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Irecv(r_arr, r_block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
          MPI_Isend(my_arr, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
          std::inplace_merge(l_arr, my_arr, my_arr + block_len);
          MPI_Isend(my_arr, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
          std::inplace_merge(my_arr, r_arr, r_arr + r_block_len);
        }
      }
    }
    // 写回数据
    memcpy(data, my_arr, block_len * sizeof(float));
    MPI_Waitall(send_rank, send_list, MPI_STATUSES_IGNORE);
    delete[] sort_arr;
  }
}