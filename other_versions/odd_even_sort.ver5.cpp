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

  if (nprocs == 1)
  {
    std::sort(data, data + n);
    return;
  }

  // 准备常量
  const int parity = rank % 2;                             // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;        // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size; // 参与排序的进程数
  const int l_proc_size = rank;                            // 左侧进程数
  const int r_proc_size = proc_size - 1 - rank;            // 右侧进程数
  const int max_swap = proc_size - 1;                      // 最大交换次数

  // 写入数据
  float *sort_arr = new float[block_size * 3];     // 排序数组
  float *my_arr = sort_arr + block_size;           // 本进程数据
  memcpy(my_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
  std::sort(my_arr, my_arr + block_len);           // 第一次排序

  // 交换数据
  MPI_Request requests[6];                                // 用于异步通信(一轮最多 6 个 Wait)
  MPI_Request *send_list = new MPI_Request[8 * max_swap]; // send_list：用于统一回收 Isend
  int send_rank = 0;
  for (int swap = 0; swap < max_swap; ++swap)
  {
    float my_min = my_arr[0];
    float my_max = my_arr[block_len - 1];
    float left_max, right_min;
    if (!l_proc_size)
    {
      MPI_Isend(&my_max, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
      MPI_Recv(&right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (my_max > right_min) // 需要交换
      {
        float *beg_pos = std::lower_bound(my_arr, my_arr + block_len, right_min);
        int r_send_len = my_arr + block_len - beg_pos;
        MPI_Isend(&r_send_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Isend(beg_pos, r_send_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        int r_recv_len;
        MPI_Recv(&r_recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_arr + block_len, r_recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::inplace_merge(beg_pos, my_arr + block_len, my_arr + block_len + r_recv_len);
      }
    }
    else if (!r_proc_size)
    {
      MPI_Isend(&my_min, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
      MPI_Recv(&left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (my_min < left_max) // 需要交换
      {
        float *end_pos = std::upper_bound(my_arr, my_arr + block_len, left_max);
        int l_send_len = end_pos - my_arr;
        MPI_Isend(&l_send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(my_arr, l_send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[2]);
        int l_recv_len;
        MPI_Recv(&l_recv_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_arr - l_recv_len, l_recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::inplace_merge(my_arr - l_recv_len, my_arr, end_pos);
      }
    }
    else
    {
      if (!parity) // 偶数进程
      {
        MPI_Isend(&my_max, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (right_min < my_max)
        {
          if (right_min < my_min)
            my_min = right_min;
          float *beg_pos = std::lower_bound(my_arr, my_arr + block_len, right_min);
          int r_send_len = my_arr + block_len - beg_pos;
          MPI_Isend(&r_send_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Isend(beg_pos, r_send_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          int r_recv_len;
          MPI_Recv(&r_recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(my_arr + block_len, r_recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::inplace_merge(beg_pos, my_arr + block_len, my_arr + block_len + r_recv_len);
        }
        MPI_Isend(&my_min, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max)
        {
          float *end_pos = std::upper_bound(my_arr, my_arr + block_len, left_max);
          int l_send_len = end_pos - my_arr;
          MPI_Isend(&l_send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Isend(my_arr, l_send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
          int l_recv_len;
          MPI_Recv(&l_recv_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(my_arr - l_recv_len, l_recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::inplace_merge(my_arr - l_recv_len, my_arr, end_pos);
        }
      }
      else // 奇数进程
      {
        MPI_Isend(&my_min, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max)
        {
          if (my_max < left_max)
            my_max = left_max;
          float *end_pos = std::upper_bound(my_arr, my_arr + block_len, left_max);
          int l_send_len = end_pos - my_arr;
          MPI_Isend(&l_send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Isend(my_arr, l_send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
          int l_recv_len;
          MPI_Recv(&l_recv_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(my_arr - l_recv_len, l_recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::inplace_merge(my_arr - l_recv_len, my_arr, end_pos);
        }
        MPI_Isend(&my_max, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (right_min < my_max)
        {
          float *beg_pos = std::lower_bound(my_arr, my_arr + block_len, right_min);
          int r_send_len = my_arr + block_len - beg_pos;
          MPI_Isend(&r_send_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Isend(beg_pos, r_send_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          int r_recv_len;
          MPI_Recv(&r_recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(my_arr + block_len, r_recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::inplace_merge(beg_pos, my_arr + block_len, my_arr + block_len + r_recv_len);
        }
      }
    }
  }

  // 写回数据
  memcpy(data, sort_arr, block_len * sizeof(float));
  MPI_Waitall(send_rank, send_list, MPI_STATUSES_IGNORE);
  delete[] sort_arr;
}