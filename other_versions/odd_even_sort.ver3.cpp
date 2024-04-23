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

  // 准备常量
  const int parity = rank % 2;                             // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;        // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size; // 参与排序的进程数
  const int l_proc_size = rank;                            // 左侧进程数
  const int r_proc_size = proc_size - 1 - rank;            // 右侧进程数
  const int max_swap = 2 * (proc_size - 1);                // 最大交换次数

  // 写入数据
  float *sort_arr = new float[block_size * 2];       // 排序数组
  memcpy(sort_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
  std::sort(sort_arr, sort_arr + block_len);         // 第一次排序

  // 交换数据
  bool no_swap[2] = {0, 0};                  // 交换记录
  int check_point = (proc_size / 3) * 4 + 1; // 检查点
  MPI_Request requests[4];                   // 用于异步通信(一轮最多 4 个 Wait)
  for (int swap = 0, stage = 0; swap < max_swap; stage = (++swap) % 2)
  {
    if (parity == stage) // 接收进程（右侧交换）
    {
      if (r_proc_size)
      {
        float my_max = sort_arr[block_len - 1];
        float right_min;
        MPI_Sendrecv(&my_max, 1, MPI_FLOAT, rank + 1, 0,
                     &right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_max > right_min) // 需要交换
        {
          float *beg_pos = std::lower_bound(sort_arr, sort_arr + block_len, right_min);
          int recv_len;
          MPI_Recv(&recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::inplace_merge(beg_pos, &sort_arr[block_len], &sort_arr[block_len + recv_len]);
          MPI_Send(&sort_arr[block_len], recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
          no_swap[stage] = false;
        }
        else
          no_swap[stage] = true;
      }
      else
        no_swap[stage] = true;
    }
    else // 发送进程（左侧交换）
    {
      if (l_proc_size)
      {
        float my_min = sort_arr[0];
        float left_max;
        MPI_Sendrecv(&my_min, 1, MPI_FLOAT, rank - 1, 0,
                     &left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max) // 需要交换
        {
          float *end_pos = std::upper_bound(sort_arr, sort_arr + block_len, left_max);
          int send_len = end_pos - sort_arr;
          MPI_Send(&send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
          MPI_Send(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
          MPI_Recv(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          no_swap[stage] = false;
        }
        else
          no_swap[stage] = true;
      }
      else
        no_swap[stage] = true;
    }
    if (swap == check_point)
    {
      bool my_flag = no_swap[0] & no_swap[1], left_flag, right_flag;
      int mi = proc_size / 2; // Reduce 和 Broadcast 的中间进程
      if (rank == mi)         // rank == mi：Reduce 和 Broadcast
      {
        if (r_proc_size) // 多于一个进程才发信息
        {
          if (!l_proc_size) // 两个进程：mi == 1
          {
            MPI_Irecv(&left_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            my_flag &= left_flag;
          }
          else // 多于两个进程：收发 4 次
          {
            MPI_Irecv(&left_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&right_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
            if (my_flag)
            {
              MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
              my_flag &= left_flag & right_flag;
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
              MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
            else
            {
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[2]);
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
              MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
            }
          }
        }
      }
      else if (rank < mi) // rank < mi：从左向右传递
      {
        if (!l_proc_size) // 第一个进程：收发 2 次
        {
          MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
          MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
        else // 收发 4 次
        {
          if (my_flag)
          {
            MPI_Recv(&left_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (my_flag &= left_flag)
            {
              MPI_Send(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Send(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD);
            }
            else
            {
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[0]);
              MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[2]);
              MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
            }
          }
          else // my_flag == false：直接发送
          {
            MPI_Irecv(&left_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
          }
        }
      }
      else if (rank > mi) // rank > mi：从右向左传递
      {
        if (!r_proc_size) // 最后一个进程：收发 2 次
        {
          if (my_flag)
          {
            MPI_Send(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          else
          {
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
          }
        }
        else // 收发 4 次
        {
          if (my_flag)
          {
            MPI_Recv(&right_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (my_flag &= right_flag)
            {
              MPI_Send(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD);
              MPI_Recv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Send(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD);
            }
            else
            {
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
              MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
              MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[2]);
              MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            }
          }
          else // my_flag == false：直接发送
          {
            MPI_Irecv(&right_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(&my_flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(&my_flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
          }
        }
      }
      // 确定是否结束
      if (my_flag)
        break;
      else
        check_point += 4;
    }
  }

  // 写回数据
  memcpy(data, sort_arr, block_len * sizeof(float));
  delete[] sort_arr;
}