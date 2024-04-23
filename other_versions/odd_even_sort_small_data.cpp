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
  const int parity = rank % 2;                                  // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;             // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size;      // 参与排序的进程数
  const int l_neighbour = rank > 0 ? rank - 1 : -1;             // 左侧邻居
  const int r_neighbour = rank < proc_size - 1 ? rank + 1 : -1; // 右侧邻居
  const int max_swap = 2 * (proc_size - 1);                     // 最大交换次数

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

  if (rank == nprocs - 1)
  {
    float *buffer = new float[n];
    MPI_Request *recv_req = new MPI_Request[nprocs - 1];
    for (int i = 0; i < nprocs - 1; ++i)
      MPI_Irecv(buffer + i * block_size, block_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &recv_req[i]);
    MPI_Waitall(nprocs - 1, recv_req, MPI_STATUSES_IGNORE);
    memcpy(buffer + (nprocs - 1) * block_size, data, block_len * sizeof(float));
    FILE *file = fopen("output", "w");
    if (file != NULL)
    {
      for (size_t i = 0; i < n; ++i)
        fprintf(file, "%f ", buffer[i]);
      fclose(file);
    }
    else
      printf("Unable to open file.\n");
  }
  else
    MPI_Send(data, block_len, MPI_FLOAT, nprocs - 1, 0, MPI_COMM_WORLD);
}