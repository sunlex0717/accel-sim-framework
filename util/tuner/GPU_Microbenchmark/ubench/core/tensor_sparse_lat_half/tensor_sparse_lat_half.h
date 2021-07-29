#ifndef LAT_TENSOR_DEF_H
#define LAT_TENSOR_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"


#define REPEAT_ITERS 4096
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

using namespace nvcuda;

// template <class T, class R>
// __global__ void tensor_latency(uint64_t *startClk, uint64_t *stopClk, T *a,
//                                T *b, R *res) {
//   int gid = blockIdx.x * blockDim.x + threadIdx.x;

//   // register T result = 0;

//   wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
//   wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
//   wmma::fragment<wmma::accumulator, 16, 16, 16, R> c_frag;

//   wmma::load_matrix_sync(a_frag, a, 16);
//   wmma::fill_fragment(c_frag, 0.0f);
//   wmma::load_matrix_sync(b_frag, b, 16);

//   // synchronize all threads
//   asm volatile("bar.sync 0;");

//   // start timing
//   uint64_t start = 0;
//   asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

//   for (int j = 0; j < REPEAT_ITERS; ++j) {
//     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//   }

//   // synchronize all threads
//   asm volatile("bar.sync 0;");

//   // stop timing
//   uint64_t stop = 0;
//   asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

//   wmma::store_matrix_sync(res, c_frag, 16, wmma::mem_row_major);

//   // write time and data back to memory
//   startClk[gid] = start;
//   stopClk[gid] = stop;
// }


__forceinline__ __device__ unsigned lane_id()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}


template <class half, class float>
__global__ void tensor_latency(uint64_t *startClk, uint64_t *stopClk, half *input_A,
                               half *input_B, uint32_t *input_E ,float *output_D, ) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /** step 0: create shared memory buffer, this step is  necessary because we need to use ldmatrix instruction which can only load from shared mem **/
   __shared__ half smem_buffer_A[MMA_M*MMA_K/2]; // wmma_m * wmma_k/2 : Sparse Matrix A
   __shared__ half smem_buffer_B[MMA_N*MMA_K]; // wmma_n * wmma_k : Dense Matrix B
   __shared__ uint32_t smem_buffer_E[MMA_M*MMA_K/16]       //e: uint32_t
  // register T result = 0;


  /** step 0.5: load from gloabl to shared **/

  if(threadIdx.x == 0){//we do not care global->shared memory when profiling tensor core, so no need for thread-cooperation
    // load A
    #pragma unroll 1
    for(int i=0;i<MMA_M*MMA_K/2;i++){
        smem_buffer_A[i] = input_A[i];
    }
    #pragma unroll 1
    for(int i =0;i<MMA_N*MMA_K;i++){
        smem_buffer_B[i] = input_B[i];
    }
    #pragma unroll 1
    for(int i=0;i<MMA_M*MMA_K/16;i++){ // think about this probably
        smem_buffer_E[i] = input_E[i];
    }

  }


    /** step 1: create register for each thread **/
  half frag_A[8]; // four b32 registrs, 8 half non-zero elements, 16 dense 
  half frag_B[8]; // four b32 registers, 8 half dense elements
  float frag_D[4]; //result(fp32) 4 f32 registers
  uint32_t frag_C; // A .b32 register containing 16 2-bit vectors to for indexing non-zero of A



  /** step 2: load data to registers via ldmatrix inst **/
  //TODO : use ldmatrix for A and B

  //TODO: cast half to 

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&frag_D[0]);
  uint32_t const E = frag_C;
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
  // wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, R> c_frag;

  // wmma::load_matrix_sync(a_frag, a, 16);
  // wmma::fill_fragment(c_frag, 0.0f);
  // wmma::load_matrix_sync(b_frag, b, 16);

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_ITERS; ++j) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // wmma::store_matrix_sync(res, c_frag, 16, wmma::mem_row_major);

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
}



template <class T, class R> float tensor_lat() {

  intilizeDeviceProp(0);

  THREADS_PER_BLOCK = 1;
  THREADS_PER_SM = 1;
  BLOCKS_NUM = 1;
  TOTAL_THREADS = 1;

  uint64_t *startClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(TOTAL_THREADS * sizeof(uint64_t));
  T *data1 = (T *)malloc(MMA_M*MMA_K/2 * sizeof(T));
  T *data2 = (T *)malloc(MMA_N*MMA_K * sizeof(T));
  R *res = (R *)malloc(TOTAL_THREADS * sizeof(R));
  uint32_t *meta_e = (uint32_t *)malloc(MMA_M*MMA_K/16 *sizeof(uint32_t) );

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  T *data1_g;
  T *data2_g;
  R *res_g;
  uint32_t *meta_e_g;

  for (uint32_t i = 0; i < MMA_M*MMA_K/2; i++) {
    data1[i] = (T)i;
    data2[i] = (T)i;
  }

  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, MMA_M*MMA_K/2 * sizeof(T)));
  gpuErrchk(cudaMalloc(&data2_g, MMA_N*MMA_K * sizeof(T)));
  gpuErrchk(cudaMalloc(&res_g, TOTAL_THREADS * sizeof(R)));
  gpuErrchk(cudaMalloc(&meta_e_g, MMA_M*MMA_K/16 *sizeof(uint32_t)));

  gpuErrchk(
      cudaMemcpy(data1_g, data1, MMA_M*MMA_K/2 * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(data2_g, data2, MMA_N*MMA_K * sizeof(T), cudaMemcpyHostToDevice));
  
  gpuErrchk(
      cudaMemcpy(meta_e_g, meta_e, MMA_M*MMA_K/16 * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // gpuErrchk(
  //     cudaMemcpy(data2_g, data2, MMA_N*MMA_K * sizeof(T), cudaMemcpyHostToDevice));

  tensor_latency<T, R><<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(
      startClk_g, stopClk_g, data1_g, data2_g, meta_e_g,res_g);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  // gpuErrchk( cudaMemcpy(res, res_g, M_SIZE*sizeof(R), cudaMemcpyDeviceToHost)
  // );

  float wmma, hmma;
  uint64_t total_time = stopClk[0] - startClk[0];
  wmma = ((float)(total_time)) / ((float)(REPEAT_ITERS));
  hmma =
      ((float)(total_time)) / ((float)(REPEAT_ITERS * SASS_hmma_per_PTX_wmma));

  std::cout << "wmma latency = " << wmma << "(clk)\n";
  std::cout << "hmma latency = " << hmma << "(clk)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return wmma;
}

#endif