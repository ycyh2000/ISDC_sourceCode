#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <cub/cub.cuh>

#define DTYPE "bfloat16"

#define M_GLOBAL 1
#define K_GLOBAL 768
#define K_GLOBAL_SPARSE 384
#define N_GLOBAL 3072


#define N 16
#define N_SHIFT 4
#define M 32
#define INDEX_REPRESENT_WIDTH 4

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K BLOCK_SIZE_N
#define BLOCK_SIZE_K_SPARSE (BLOCK_SIZE_N / 2)

#define THREAD_SIZE_M 32
#define THREAD_SIZE_N 1

#define PROCESSLINE 1
#define BLOCKSIZE_ 32


#define MANT_HEAD_DATA_WIDTH 7
#define MANT_DECOMPRESS_NUM_PER 4 //This value is a fixed constant equal to 4.
#define MANT_DECOMPRESS_NUM_PER_SHIFT 2 //Shift right by this number of bits to divide by MANT_DECOMPRESS_NUM_PER (since 4 = 2^2)

// REGNUM must be configured such that (REGNUM + 1) * 32 equals a power of two.
#define INDEX_REGNUM 3
#define WIDTH_REGNUM 3
#define MANT_REGNUM 7

#define INDEX_MEMORYSIZE (INDEX_REGNUM) * 32 * 32
#define WIDTH_MEMORYSIZE (WIDTH_REGNUM) * 32 * 32
#define MANT_MEMORYSIZE (MANT_REGNUM) * 32 * 32

const uint32_t INDEX_MEMORY_MASK = ((INDEX_REGNUM + 1) * 32) - 1;
const uint32_t WIDTH_MEMORY_MASK = ((WIDTH_REGNUM + 1) * 32) - 1;
const uint32_t MANT_MEMORY_MASK = ((MANT_REGNUM + 1) * 32) - 1;


#define REGNUM 3
#define MEMORYSIZE (REGNUM) * 32 * 32
#define GROUPNUM 1  //Process a batch of data, referring to the N elements in N:M sparse compression.

const uint32_t MEMORY_MASK = ((REGNUM + 1) * 32) - 1;


#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void decompress_kernel(
    uint32_t* compressed_data_width, int compressed_data_width_size,
    uint32_t* compressed_weight_data, int compressed_weight_data_size,
    uint32_t* compressed_index_data, int compressed_index_data_size,
    uint32_t* mantDataStartAddr,
    int data_width_len,
    int base_width,

    uint32_t* decompressed_index_gpu,
    uint32_t* decompressed_weight_gpu
)

{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int data_width_len_reg = data_width_len;


    __shared__ uint32_t MemoryMessage[32];
    MemoryMessage[tid] = 0;

    uint32_t dataWidthBaseAddr = bid * PROCESSLINE * (K_GLOBAL_SPARSE / N) * data_width_len_reg;
    __shared__ uint32_t dataWidthMemory[(WIDTH_REGNUM + 1) * 32];
    uint32_t currentDataWidthSharedWriteAddr = tid * 32;
    uint32_t currentDataWidthSharedReadAddr = tid * GROUPNUM * data_width_len_reg;
    uint32_t dataWidthSharedWriteAddr = 0;  //MemoryMessage[4]
    uint32_t dataWidthSharedReadAddr = 0;   //MemoryMessage[5]
    float dataWidthReadLoop = (PROCESSLINE * (K_GLOBAL_SPARSE / N) * data_width_len_reg) / (32 * 32);


    uint32_t mantDataBaseAddr = mantDataStartAddr[bid * PROCESSLINE];
    uint32_t mantDataEndAddr = mantDataStartAddr[bid * PROCESSLINE + 1];
    __shared__ uint32_t mantDataMemory[(MANT_REGNUM + 1) * 32];
    uint32_t currentDataSharedWriteAddr = tid * 32;
    uint32_t currentDataSharedReadAddr = mantDataBaseAddr % 32;
    uint32_t dataSharedWriteAddr = 0;   //MemoryMessage[8]
    uint32_t dataSharedReadAddr = 0;    //MemoryMessage[9]
    uint32_t dataRegEmpty = MANT_REGNUM * 32 * 32 - (dataSharedWriteAddr - dataSharedReadAddr);
    float dataReadLoop = (mantDataEndAddr - mantDataBaseAddr) / (32 * 32);
    uint32_t dataDecompressedRound = 0;


    uint32_t startAddr;
    uint32_t endAddr;
    uint32_t endWordAddr;
    uint32_t startWordAddr;
    uint32_t endBitOffset;
    uint32_t endWord;
    uint32_t startWord;


    __shared__ uint32_t scan_width[7][64];  // 6 layers, 64 elements per layer
    __shared__ uint32_t scan_mant[6][64];
    uint32_t flag = 0;  //The first bit indicates index read done, the second bit indicates data width read done, the third bit indicates data read done.



    for (int i = 0; i < ((K_GLOBAL * PROCESSLINE) / M) >> 5; i++)
    {
        //if ((bid == 0) && (tid == 0)) {
        //    printf("tid = %d, i = %d\n", tid, i);
        //}
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS DATA WIDTH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS DATA WIDTH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS DATA WIDTH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS DATA WIDTH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS DATA WIDTH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //read data width
        if ((dataWidthReadLoop >= 0) && (WIDTH_MEMORYSIZE - (dataWidthSharedWriteAddr - dataWidthSharedReadAddr) >= 32 * 32)) {
            dataWidthMemory[(currentDataWidthSharedWriteAddr >> 5) & WIDTH_MEMORY_MASK] = compressed_data_width[(dataWidthBaseAddr + currentDataWidthSharedWriteAddr) >> 5];

            dataWidthSharedWriteAddr = 32 * 32 + dataWidthSharedWriteAddr;

            currentDataWidthSharedWriteAddr = currentDataWidthSharedWriteAddr + 32 * 32;
            if (currentDataWidthSharedWriteAddr >= (((K_GLOBAL_SPARSE / N) * data_width_len_reg * PROCESSLINE + 31) & ~31) - 32) {
                currentDataWidthSharedWriteAddr = (((K_GLOBAL_SPARSE / N) * data_width_len_reg * PROCESSLINE + 31) & ~31) - 32;
            }
            dataWidthReadLoop = dataWidthReadLoop - 1;
        }
        //******************************
        //read data width from global memory and concat
        //******************************
        startAddr = currentDataWidthSharedReadAddr;
        endAddr = (currentDataWidthSharedReadAddr + GROUPNUM * data_width_len_reg - 1);
        endWordAddr = (endAddr >> 5) & WIDTH_MEMORY_MASK;
        startWordAddr = (startAddr >> 5) & WIDTH_MEMORY_MASK;
        endBitOffset = endAddr & 0x1F;
        //read data
        endWord = dataWidthMemory[endWordAddr];
        startWord = dataWidthMemory[startWordAddr];
        // concat
        uint32_t temp_data_concat = (endWord >> (31 - endBitOffset)) | (startWord << (endBitOffset + 1));
        uint32_t temp_width = temp_data_concat & ((uint64_t)((1U << data_width_len_reg) - 1));
        temp_width = temp_width + base_width;

        //if ((bid == 0) && (i <= 1))
        //    printf("tid = %d, width = %u\n", tid, temp_width - base_width);

        //1.calculate data len
        scan_width[0][tid] = 0;
        scan_width[1][tid] = 0;
        scan_width[2][tid] = 0;
        scan_width[3][tid] = 0;
        scan_width[4][tid] = 0;
        scan_width[5][tid] = 0;

        //int temp0, temp1;
        // Store input value and initialize temp0
        scan_width[0][32 + tid] = (temp_width * (N - 1) + MANT_HEAD_DATA_WIDTH);
        int temp0 = (temp_width * (N - 1) + MANT_HEAD_DATA_WIDTH);
        // Layer 1: Stride 1 accumulation
        int temp1 = scan_width[0][32 + tid - 1];
        temp0 = temp0 + temp1;
        scan_width[1][32 + tid] = temp0;
        // Layer 2: Stride 2 accumulation
        temp1 = scan_width[1][32 + tid - 2];
        temp0 = temp0 + temp1;
        scan_width[2][32 + tid] = temp0;
        // Layer 3: Stride 4 accumulation
        temp1 = scan_width[2][32 + tid - 4];
        temp0 = temp0 + temp1;
        scan_width[3][32 + tid] = temp0;
        // Layer 4: Stride 8 accumulation
        temp1 = scan_width[3][32 + tid - 8];
        temp0 = temp0 + temp1;
        scan_width[4][32 + tid] = temp0;
        // Layer 5: Stride 16 accumulation
        temp1 = scan_width[4][32 + tid - 16];
        temp0 = temp0 + temp1;
        scan_width[5][32 + tid] = temp0;
        //Layer 6: Residual Width for each N:M group
        scan_width[6][32 + tid] = temp_width;

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS MANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS MANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS MANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS MANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PROCESS MANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        uint32_t completed_group = 0;
        uint32_t current_group = ((tid << MANT_DECOMPRESS_NUM_PER_SHIFT) >> N_SHIFT);
        uint32_t backID_ele = tid % (N / MANT_DECOMPRESS_NUM_PER);
        for (int j = 0; j < (N / MANT_DECOMPRESS_NUM_PER); j++) {
            //read mant data
            if ((dataReadLoop >= 0) && (dataRegEmpty >= 32 * 32)) {
                mantDataMemory[(currentDataSharedWriteAddr >> 5) & MANT_MEMORY_MASK] = compressed_weight_data[(mantDataBaseAddr + currentDataSharedWriteAddr) >> 5];

                dataSharedWriteAddr = 32 * 32 + dataSharedWriteAddr;

                currentDataSharedWriteAddr = currentDataSharedWriteAddr + 32 * 32;
                if (currentDataSharedWriteAddr >= mantDataEndAddr) {
                    currentDataSharedWriteAddr = currentDataSharedWriteAddr - 32;
                }
                dataRegEmpty = MANT_REGNUM * 32 * 32 - (dataSharedWriteAddr - dataSharedReadAddr);

                dataReadLoop = dataReadLoop - 1;
            }

            //******************************
            //read data from global memory and concat
            //******************************
            uint32_t packedDatax, packedDatay, packedDataz, packedDataw;
            uint32_t currentMantGroupAddr = scan_width[5][32 + completed_group + current_group];
            uint32_t allMantGroupAddr = scan_width[5][32 + completed_group + ((31 << MANT_DECOMPRESS_NUM_PER_SHIFT) >> N_SHIFT)];
            uint32_t currentGroupResidualWidth = scan_width[6][32 + completed_group + current_group];
            uint32_t backID = (N / MANT_DECOMPRESS_NUM_PER) - (backID_ele)-1;
            completed_group = completed_group + (32 * MANT_DECOMPRESS_NUM_PER) / N;

            endAddr = (currentDataSharedReadAddr + currentMantGroupAddr - 1 - backID * MANT_DECOMPRESS_NUM_PER * currentGroupResidualWidth);
            if (backID_ele == 0) {
                startAddr = endAddr - (MANT_DECOMPRESS_NUM_PER - 1) * currentGroupResidualWidth - MANT_HEAD_DATA_WIDTH;
            }
            else {
                startAddr = endAddr - MANT_DECOMPRESS_NUM_PER * currentGroupResidualWidth;
            }

            dataSharedReadAddr = currentDataSharedReadAddr + allMantGroupAddr;

            endWordAddr = (endAddr >> 5) & MANT_MEMORY_MASK;
            startWordAddr = (startAddr >> 5) & MANT_MEMORY_MASK;
            endBitOffset = endAddr & 0x1F;
            //read data
            endWord = mantDataMemory[endWordAddr];
            startWord = mantDataMemory[startWordAddr];
            // concat
            uint64_t mant_data_concat = (endWord >> (31 - endBitOffset)) | (startWord << (endBitOffset + 1));
            //if ((bid == 0) && (i <= 0) && (j == 0)) {
            //    printf("tid = %d, mant_data_concat = %llu, endAddr = %u\n", tid, mant_data_concat, endAddr);
            //}
            packedDataw = mant_data_concat & ((1U << currentGroupResidualWidth) - 1);
            mant_data_concat = (mant_data_concat >> currentGroupResidualWidth);
            packedDataz = mant_data_concat & ((1U << currentGroupResidualWidth) - 1);
            mant_data_concat = (mant_data_concat >> currentGroupResidualWidth);
            packedDatay = mant_data_concat & ((1U << currentGroupResidualWidth) - 1);
            mant_data_concat = (mant_data_concat >> currentGroupResidualWidth);
            if (backID_ele == 0) {
                packedDatax = mant_data_concat & ((1U << MANT_HEAD_DATA_WIDTH) - 1);
            }
            else {
                packedDatax = mant_data_concat & ((1U << currentGroupResidualWidth) - 1);
            }
            packedDatay = packedDatay + packedDatax;
            packedDataz = packedDataz + packedDatay;
            packedDataw = packedDataw + packedDataz;


            //1. calculate data length
            scan_mant[0][tid] = 0;
            scan_mant[1][tid] = 0;
            scan_mant[2][tid] = 0;
            scan_mant[3][tid] = 0;
            scan_mant[4][tid] = 0;
            scan_mant[5][tid] = 0;
            //// Store input value and initialize temp0
            scan_mant[0][32 + tid] = packedDataw;
            temp0 = packedDataw;
            // Layer 1: Stride 1 accumulation
            int temp1 = scan_mant[0][32 + tid - 1];
            temp0 = temp0 + temp1;
            scan_mant[1][32 + tid] = temp0;
            // Layer 2: Stride 2 accumulation
            temp1 = scan_mant[1][32 + tid - 2];
            temp0 = temp0 + temp1;
            scan_mant[2][32 + tid] = temp0;
            // Layer 3: Stride 4 accumulation
            temp1 = scan_mant[2][32 + tid - 4];
            temp0 = temp0 + temp1;
            scan_mant[3][32 + tid] = temp0;
            // Layer 4: Stride 8 accumulation
            temp1 = scan_mant[3][32 + tid - 8];
            temp0 = temp0 + temp1;
            scan_mant[4][32 + tid] = temp0;
            // Layer 5: Stride 16 accumulation
            temp1 = scan_mant[4][32 + tid - 16];
            temp0 = temp0 + temp1;
            scan_mant[5][32 + tid] = temp0;
            if (backID_ele != 0) {
                uint32_t packedDataTemp = scan_mant[5][32 + tid - 1] - scan_mant[5][32 + ((tid>>2)<<2) - 1];
                packedDatax = packedDatax + packedDataTemp;
                packedDatay = packedDatay + packedDataTemp;
                packedDataz = packedDataz + packedDataTemp;
                packedDataw = packedDataw + packedDataTemp;
            }


            //////1. calculate data length
            //scan_mant[0][tid] = 0;
            //scan_mant[1][tid] = 0;
            ////// Store input value and initialize temp0
            //scan_mant[0][32 + tid] = packedDataw;
            //temp0 = packedDataw;
            //// Layer 1: Accumulate within group with step 1
            //temp1 = (tid % 4 == 0) ? 0 : scan_mant[0][32 + tid - 1];
            //temp0 = temp0 + temp1;
            //scan_mant[1][32 + tid] = temp0;
            //// Layer 2: Accumulate within group with step 2
            //temp1 = (tid % 4 < 2) ? 0 : scan_mant[1][32 + tid - 2];
            //temp0 = temp0 + temp1;
            //scan_mant[2][32 + tid] = temp0;
            //// Layer 3: Accumulate within group with step 4 (actually step 3, but using step 4 to cross group boundary)
            //temp1 = (tid % 4 == 3) ? scan_mant[2][32 + tid - 3] : 0;
            //temp0 = temp0 + temp1;
            //scan_mant[3][32 + tid] = temp0;
            //if (backID_ele == 0)
            //{
            //    packedDatax = packedDatax;
            //    packedDatay = packedDatay;
            //    packedDataz = packedDataz;
            //    packedDataw = packedDataw;
            //}
            //else {
            //    uint32_t packedDataTemp = scan_mant[3][32 + tid - 1];
            //    packedDatax = packedDatax + packedDataTemp;
            //    packedDatay = packedDatay + packedDataTemp;
            //    packedDataz = packedDataz + packedDataTemp;
            //    packedDataw = packedDataw + packedDataTemp;
            //}

            dataRegEmpty = MANT_REGNUM * 32 * 32 - (dataSharedWriteAddr - dataSharedReadAddr);

            (decompressed_weight_gpu)[(K_GLOBAL_SPARSE * bid * PROCESSLINE + dataDecompressedRound * 32 * GROUPNUM * MANT_DECOMPRESS_NUM_PER + tid * MANT_DECOMPRESS_NUM_PER) + 0] = packedDatax;
            (decompressed_weight_gpu)[(K_GLOBAL_SPARSE * bid * PROCESSLINE + dataDecompressedRound * 32 * GROUPNUM * MANT_DECOMPRESS_NUM_PER + tid * MANT_DECOMPRESS_NUM_PER) + 1] = packedDatay;
            (decompressed_weight_gpu)[(K_GLOBAL_SPARSE * bid * PROCESSLINE + dataDecompressedRound * 32 * GROUPNUM * MANT_DECOMPRESS_NUM_PER + tid * MANT_DECOMPRESS_NUM_PER) + 2] = packedDataz;
            (decompressed_weight_gpu)[(K_GLOBAL_SPARSE * bid * PROCESSLINE + dataDecompressedRound * 32 * GROUPNUM * MANT_DECOMPRESS_NUM_PER + tid * MANT_DECOMPRESS_NUM_PER) + 3] = packedDataw;

            dataDecompressedRound++;
        }

        currentDataSharedReadAddr = dataSharedReadAddr;


        dataWidthSharedReadAddr = dataWidthSharedReadAddr + 32 * GROUPNUM * data_width_len_reg;
        currentDataWidthSharedReadAddr = currentDataWidthSharedReadAddr + 32 * GROUPNUM * data_width_len_reg;
    }

}


extern "C" {

        void call_decompress_kernel(uint32_t* compressed_data_width, int compressed_data_width_size,
            uint32_t* compressed_weight_data, int compressed_weight_data_size,
            uint32_t* compressed_index_data, int compressed_index_data_size,
            uint32_t* startAddr,
            int oc_size, int ic_size,
            int data_width_len,
            int base_width
        ) {
        const int h = N_GLOBAL;
        const int vecNum = K_GLOBAL;
        int w = int(vecNum / M) * N;
        const int minibatch = M_GLOBAL;

        // set up device
        int dev = 0;
        cudaSetDevice(dev);

        uint32_t* compressed_data_width_gpu;
        uint32_t* compressed_weight_data_gpu;
        uint32_t* compressed_index_data_gpu;
        uint32_t* startAddr_gpu;
        uint32_t* decompressed_index_gpu;
        uint32_t* decompressed_weight_gpu;
        float* g_vec, * g_result, * gpuRef;

        uint32_t* decompressed_index = (uint32_t*)malloc(N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t));
        uint32_t* decompressed_weight = (uint32_t*)malloc(N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t));

        CUDA_CHECK(cudaMalloc((void**)&compressed_data_width_gpu, compressed_data_width_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&compressed_weight_data_gpu, compressed_weight_data_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&compressed_index_data_gpu, compressed_index_data_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&startAddr_gpu, (N_GLOBAL + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&decompressed_index_gpu, N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&decompressed_weight_gpu, N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(compressed_data_width_gpu, compressed_data_width, compressed_data_width_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(compressed_weight_data_gpu, compressed_weight_data, compressed_weight_data_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(compressed_index_data_gpu, compressed_index_data, compressed_index_data_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(startAddr_gpu, startAddr, (N_GLOBAL + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

        //Warm up the kernel
        int ntimes = 100;

        dim3 dimBlock(BLOCKSIZE_);
        dim3 dimGrid(oc_size / PROCESSLINE);

        for (int i = 0; i < ntimes; i += 1) {
            decompress_kernel << < dimGrid, dimBlock >> > (
                compressed_data_width_gpu, compressed_data_width_size,
                compressed_weight_data_gpu, compressed_weight_data_size,
                compressed_index_data_gpu, compressed_index_data_size,
                startAddr_gpu,
                data_width_len,
                base_width,
                decompressed_index_gpu,
                decompressed_weight_gpu
                );
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < ntimes; i += 1) {
            decompress_kernel << < dimGrid, dimBlock >> > (
                compressed_data_width_gpu, compressed_data_width_size,
                compressed_weight_data_gpu, compressed_weight_data_size,
                compressed_index_data_gpu, compressed_index_data_size,
                startAddr_gpu,
                data_width_len,
                base_width,
                decompressed_index_gpu,
                decompressed_weight_gpu
                );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        printf("kernel runtime is %lf ms\n", elapsed_time / ntimes);

        cudaError_t kernelError = cudaGetLastError();
        if (kernelError != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(kernelError));
            return;
        }

        CUDA_CHECK(cudaMemcpy(decompressed_index, decompressed_index_gpu, N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(decompressed_weight, decompressed_weight_gpu, N_GLOBAL * K_GLOBAL_SPARSE * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        printf("start\n");
        size_t start_elements = (N_GLOBAL - 1) * K_GLOBAL_SPARSE;
        size_t total_elements = (N_GLOBAL)*K_GLOBAL_SPARSE;
        for (size_t i = start_elements; i < total_elements; i++) {
            printf("decompressed_weight[%zu] = %u\n", i, decompressed_weight[i]);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(compressed_data_width_gpu));
        CUDA_CHECK(cudaFree(compressed_weight_data_gpu));
        CUDA_CHECK(cudaFree(compressed_index_data_gpu));
        CUDA_CHECK(cudaDeviceReset());
    }

}
