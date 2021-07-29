#ifndef LAT_TENSOR_DEF_H
#define LAT_TENSOR_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include "../../../hw_def/hw_def.h"


#define REPEAT_ITERS 4096
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

// note this is just fake meta data, used for performance microbenchmarking
void initialize_fake_metadata_2_4(uint32_t* metadata, int row_size,int col_size){
  int range = 6;
  uint32_t FourToTwoMeta[6] = {0x4, 0x8, 0x9, 0xc, 0xd, 0xe};
  for(int i = 0; i < row_size * col_size/16;i++){ // 32 bit can represent 16 indexes , each index has 2 bit
      uint32_t result = 0x0;
      for (int n = 0; n < 32 / 4; ++n) {
        double rnd = double(std::rand()) / double(RAND_MAX);
        rnd = range * rnd;
        uint32_t meta = FourToTwoMeta[(int)rnd];

        result = (uint32_t)(result | ((uint32_t)(meta << (i * 4))));
    }
    metadata[i] = result;
  }

   // convert to meta data
  //  print_matrix<int>(temps_arr,row_size,col_size,layout);
}

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

template <typename T, typename R>
__global__ void tensor_latency(uint64_t *startClk, uint64_t *stopClk, T *input_A,
                               T *input_B, uint32_t *input_E ,R *output_D ){

  printf("no implementattion");
  assert(0);

}



template <>
__global__ void tensor_latency<half,half>(uint64_t *startClk, uint64_t *stopClk, half *input_A,
                               half *input_B, uint32_t *input_E ,half *output_D ) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /** step 0: create shared memory buffer, this step is  necessary because we need to use ldmatrix instruction which can only load from shared mem **/
   __shared__ half smem_buffer_A[MMA_M*MMA_K/2]; // wmma_m * wmma_k/2 : Sparse Matrix A
   __shared__ half smem_buffer_B[MMA_N*MMA_K]; // wmma_n * wmma_k : Dense Matrix B
   __shared__ uint32_t smem_buffer_E[MMA_M*MMA_K/16];       //e: uint32_t
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
  half frag_A[8]; // four b32 registers, 8 half non-zero elements, 16 dense 
  half frag_B[8]; // four b32 registers, 8 half dense elements
  half frag_D[4]; //result(half) two fp16x2 registers
  uint32_t frag_C; // A .b32 register containing 16 2-bit vectors to for indexing non-zero of A

  /** step 2: load data to registers via ldmatrix inst **/
  //TODO : use ldmatrix for A and B correctly
  // Note for test, we just use some random values,we do not care correctness at this moment
  //#pragma unroll 1
  // fake load
  for(int i = 0;i<8 ;i++){
    frag_A[i] = smem_buffer_A[i+lane_id()];
    frag_B[i] = smem_buffer_B[i+lane_id()];
  }

  //TODO: cast half to 
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  uint32_t *C = reinterpret_cast<uint32_t *>(&frag_D[0]);
  uint32_t *D = C; 
  uint32_t const E = frag_C;


  //warm-up
  asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), 
          "r"(E));
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int j = 0; j < REPEAT_ITERS; ++j) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "r"(C[0]), "r"(C[1]), 
          "r"(E));
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  // fake store,
  half *res = reinterpret_cast<half *>(D);
  for(int i =0; i<4;i++){
    output_D[i+lane_id()] = res[i];
  }

  // printf("I am tread %d",lane_id());
  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
}



template <>
__global__ void tensor_latency<half,float>(uint64_t *startClk, uint64_t *stopClk, half *input_A,
                               half *input_B, uint32_t *input_E ,float *output_D ) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  /** step 0: create shared memory buffer, this step is  necessary because we need to use ldmatrix instruction which can only load from shared mem **/
   __shared__ half smem_buffer_A[MMA_M*MMA_K/2]; // wmma_m * wmma_k/2 : Sparse Matrix A
   __shared__ half smem_buffer_B[MMA_N*MMA_K]; // wmma_n * wmma_k : Dense Matrix B
   __shared__ uint32_t smem_buffer_E[MMA_M*MMA_K/16];       //e: uint32_t
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
  //TODO : use ldmatrix for A and B correctly
  // Note for test, we just use some random values,we do not care correctness at this moment
  //#pragma unroll 1
  // fake load
  for(int i = 0;i<8 ;i++){
    frag_A[i] = smem_buffer_A[i+lane_id()];
    frag_B[i] = smem_buffer_B[i+lane_id()];
  }

  //TODO: cast half to 
  uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
  float *C = reinterpret_cast<float *>(&frag_D[0]);
  float *D = C; 
  uint32_t const E = frag_C;

  //warm up
  asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(B[2]), "r"(B[3]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(E));
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  uint64_t stop = 0;
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
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  // fake store,
  for(int i =0; i<4;i++){
    output_D[i+lane_id()] = D[i];
  }

  // printf("I am tread %d",lane_id());
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
  R *res = (R *)malloc(MMA_M*MMA_N * sizeof(R));
  uint32_t *meta_e = (uint32_t *)malloc(MMA_M*MMA_K/16 *sizeof(uint32_t) );

  initialize_fake_metadata_2_4(meta_e,MMA_M,MMA_K);

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
  gpuErrchk(cudaMalloc(&res_g, MMA_M*MMA_N * sizeof(R)));
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

  float mma;
  uint64_t total_time = stopClk[0] - startClk[0];
  mma =
      ((float)(total_time)) / ((float)(REPEAT_ITERS));

  std::cout << "mma latency = " << mma << "(clk)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  return mma;
}

#endif