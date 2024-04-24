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

  // 数据量小，平衡通信开销的算法
  if (n < 20000000)
  {
    const int max_swap = proc_size - 1; // 合并两轮交换做优化

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
    MPI_Request recv_req;                                   // 单独 wait Irecv
    MPI_Status recv_status;                                 // 单独回收 Irecv

    if (l_neighbour == -1) // 第一个进程
      for (int swap = 0; swap < max_swap; ++swap)
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
    else if (r_neighbour == -1) // 最后一个进程
      for (int swap = 0; swap < max_swap; ++swap)
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
    else // 中间进程
    {
      if (parity == 0) // 偶数进程
        for (int swap = 0; swap < max_swap; ++swap)
        {
          float my_max = *(my_arr_end - 1);
          float right_min;
          MPI_Isend(&my_max, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&right_min, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_max > right_min) // 需要交换
          {
            float my_min = right_min < *my_arr_beg ? right_min : *my_arr_beg;
            float left_max;
            MPI_Isend(&my_min, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
            MPI_Irecv(&left_max, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &recv_req);
            float *beg_pos = std::upper_bound(my_arr_beg, my_arr_end, right_min);
            int send_len = my_arr_end - beg_pos;
            MPI_Sendrecv(beg_pos, send_len, MPI_FLOAT, r_neighbour, 0,
                         my_arr_end, block_size, MPI_FLOAT, r_neighbour, 0,
                         MPI_COMM_WORLD, &recv_status);
            int recv_len;
            MPI_Get_count(&recv_status, MPI_FLOAT, &recv_len);
            std::inplace_merge(beg_pos, my_arr_end, my_arr_end + recv_len);
            MPI_Wait(&recv_req, &recv_status);
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
          else // 直接第二轮
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
      else // 奇数进程
        for (int swap = 0; swap < max_swap; ++swap)
        {
          float my_min = *my_arr_beg;
          float left_max;
          MPI_Isend(&my_min, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Recv(&left_max, 1, MPI_FLOAT, l_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (my_min < left_max) // 需要交换
          {
            float my_max = left_max > *(my_arr_end - 1) ? left_max : *(my_arr_end - 1);
            float right_min;
            MPI_Isend(&my_max, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
            MPI_Irecv(&right_min, 1, MPI_FLOAT, r_neighbour, 0, MPI_COMM_WORLD, &recv_req);
            float *end_pos = std::lower_bound(my_arr_beg, my_arr_end, left_max);
            int send_len = end_pos - my_arr_beg;
            MPI_Sendrecv(my_arr_beg, send_len, MPI_FLOAT, l_neighbour, 0,
                         l_recv_arr, block_size, MPI_FLOAT, l_neighbour, 0,
                         MPI_COMM_WORLD, &recv_status);
            int recv_len;
            MPI_Get_count(&recv_status, MPI_FLOAT, &recv_len);
            std::merge(my_arr_beg, end_pos, l_recv_arr, l_recv_arr + recv_len, &sort_arr[block_size - recv_len]);
            MPI_Wait(&recv_req, &recv_status);
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
          else // 直接开启第二轮
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
    const int max_swap = (proc_size - 1) * 2;

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