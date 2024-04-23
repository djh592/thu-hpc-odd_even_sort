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
    FILE *file = fopen("stdoutput", "w");
    if (file != NULL)
    {
      for (size_t i = 0; i < n; ++i)
        fprintf(file, "%f ", data[i]);
      fclose(file);
    }
    else
      printf("Unable to open file.\n");
    return;
  }

  // 准备常量
  const int parity = rank % 2;                             // 进程奇偶性
  const int block_size = (n + nprocs - 1) / nprocs;        // 每个进程的数据长度
  const int proc_size = (n + block_size - 1) / block_size; // 参与排序的进程数
  const size_t l_block_len = rank ? block_size : 0;        // 左侧数据长度
  const size_t r_block_len = rank < proc_size - 2 ? block_size : rank == proc_size - 2 ? n - (proc_size - 1) * block_size
                                                                                       : 0; // 右侧数据长度
  const int max_swap = 2 * (proc_size - 1);                                                 // 最大交换次数

  // 写入数据
  float *sort_arr = new float[l_block_len + block_len + r_block_len]; // 排序数组
  float *my_arr = sort_arr + l_block_len;                             // 本进程数据
  float *my_arr_end = my_arr + block_len;
  memcpy(my_arr, data, block_len * sizeof(float)); // 将数据写入排序数组
  std::sort(my_arr, my_arr + block_len);           // 第一次排序

  // 交换数据
  int send_rank = 0;
  MPI_Request *send_list = new MPI_Request[3 * max_swap]; // send_list：用于统一回收 Isend
  for (int swap = 0, stage = 0; swap < max_swap; stage = (++swap) % 2)
  {
    if (parity == stage) // 接收进程（右侧交换）
    {
      if (r_block_len)
      {
        float my_max = my_arr[block_len - 1];
        float right_min;
        MPI_Isend(&my_max, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&right_min, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_max > right_min) // 需要交换
        {
          float *beg_pos = std::lower_bound(my_arr, my_arr + block_len, right_min);
          int send_len = my_arr_end - beg_pos;
          MPI_Isend(&send_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Isend(beg_pos, send_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          int recv_len;
          MPI_Recv(&recv_len, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          float *recv_pos = my_arr_end;
          MPI_Recv(recv_pos, recv_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (*(my_arr_end) != right_min)
          {
            printf("swap %d rank %d: arr[end]: %f, right_min: %f\n", swap, rank, *(my_arr_end), right_min);
          }
          std::inplace_merge(beg_pos, my_arr_end, my_arr_end + recv_len);
        }
      }
    }
    else // 发送进程（左侧交换）
    {
      if (l_block_len)
      {
        float my_min = my_arr[0];
        float left_max;
        MPI_Isend(&my_min, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
        MPI_Recv(&left_max, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (my_min < left_max) // 需要交换
        {
          float *end_pos = std::upper_bound(my_arr, my_arr + block_len, left_max);
          int send_len = end_pos - my_arr;
          MPI_Isend(&send_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          MPI_Isend(my_arr, send_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &send_list[send_rank++]);
          int recv_len;
          MPI_Recv(&recv_len, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          float *recv_pos = my_arr - recv_len;
          MPI_Recv(recv_pos, recv_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          if (*(my_arr - 1) != left_max)
          {
            printf("swap %d rank %d: arr[-1]: %f, left_max: %f\n", swap, rank, *(my_arr - 1), left_max);
          }
          std::inplace_merge(recv_pos, my_arr, end_pos);
        }
      }
    }
  }

  // 写回数据
  memcpy(data, my_arr, block_len * sizeof(float));
  MPI_Waitall(send_rank, send_list, MPI_STATUSES_IGNORE);
  delete[] sort_arr;

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