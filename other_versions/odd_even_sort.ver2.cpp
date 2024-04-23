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
  bool first_rank = rank == 0;
  // 准备数据
  float *sort_arr = new float[((n + nprocs - 1) / nprocs) * 2]; // 排序数组
  memcpy(sort_arr, data, block_len * sizeof(float));            // 将数据写入排序数组
  std::sort(sort_arr, sort_arr + block_len);                    // 第一次排序

  // const size_t r_block_len = last_rank ? 0 : rank == nprocs - 2 ? n - (rank + 1) * block_len
  //                                                               : block_len; // 右侧数据长度

  // 交换数据
  bool no_swap[2] = {0, 0};                                                                           // 交换记录
  MPI_Request requests[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL}; // 用于异步通信(一轮最多 4 个 Wait)
  for (int stage = 0, odd_even = 0;; odd_even = (++stage) % 2)                                        // odd_even 0: 偶数阶段，odd_even 1: 奇数阶段
  {
    if (rank % 2 == odd_even) // 接收进程（排序进程）
    {
      if (!last_rank) // 不在最后一个进程：接收右侧数据
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
          no_swap[odd_even] = false;
        }
        else // 不需要交换
          no_swap[odd_even] = true;
      }
      else // 最后一个进程：不需要交换
        no_swap[odd_even] = true;
    }
    else // 发送进程（提供数据的进程）
    {
      if (!first_rank) // 不在第一个进程：发送左侧数据
      {
        float my_min = sort_arr[0];
        float left_max;
        MPI_Sendrecv(&my_min, 1, MPI_FLOAT, rank - 1, 0,
                     &left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max) // 需要交换
        {
          float *end_pos = std::upper_bound(sort_arr, sort_arr + block_len, left_max); // 查找排序结束的位置
          int send_len = end_pos - sort_arr;
          MPI_Send(&send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);                            // 发送待排序数据长度
          MPI_Send(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);                    // 发送待排序数据
          MPI_Recv(sort_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // 接收新数据
          no_swap[odd_even] = false;
        }
        else // 不需要交换
          no_swap[odd_even] = true;
      }
      else // 第一个进程：不需要交换
        no_swap[odd_even] = true;
    }
    if (stage % 4) // 4 轮检查一次
    {
      bool my_flag = no_swap[0] & no_swap[1], left_flag, right_flag;
      int mi = nprocs / 2; // Reduce 和 Broadcast 的中间进程
      if (rank == mi)      // rank == mi：Reduce 和 Broadcast
      {
        if (!first_rank) // 多于一个进程才发信息
        {
          if (last_rank) // 两个进程：mi == 1
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
        if (first_rank) // 第一个进程：收发 2 次
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
        if (last_rank) // 最后一个进程：收发 2 次
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
    }
  }
  // 写回数据
  memcpy(data, sort_arr, block_len * sizeof(float));
  delete[] sort_arr;
}