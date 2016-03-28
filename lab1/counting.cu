#include "counting.h"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

/* brief index leaf node
*/
__global__ void IndexLeafNode(const char *text, bool *forest, int text_size, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.x*step+blockDim.x;
    forest[offset+threadIdx.x] = (text[idx] != '\n' && idx < text_size);
}

/* brief index interal node
*/
__global__ void IndexInteranlNode(bool *forest, int base, int step)
{
    int left  = 2*(base+threadIdx.x);
    int right = left + 1;
    int offset = blockIdx.x*step;
    forest[offset+base+threadIdx.x] = (forest[offset+left]&&forest[offset+right]);
}

/* traverse the binary indexed tree and find position
*/
__global__ void FindPos(int *pos, bool *forest, int text_size, int order, int step)
{
    int text_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.x*step;
    if(text_idx < text_size) {
        if(!forest[offset+blockDim.x+threadIdx.x]) {
            pos[text_idx] = 0;
        } else {
            bool isCurBlock = true;
            bool isLeftMost = (blockIdx.x < 1);
            int  nodeIdx    = blockDim.x+threadIdx.x;
            int  leftBound  = blockDim.x;
            int  rightBound = 2*blockDim.x-1;
            int  alignOrder = 0;
            // bottom-up
            while(alignOrder != order) {
                int leftInx;
                if(nodeIdx-1 < leftBound) {
                    if(isLeftMost) break;
                    isCurBlock = false;
                    leftInx = offset-step+rightBound;
                } else {
                    leftInx = offset+nodeIdx-1;
                }
                
                if(!forest[leftInx]) break;
                
                rightBound = leftBound-1;
                leftBound /= 2;
                nodeIdx /= 2;
                alignOrder++;
            }
            
            // top-down
            if(alignOrder == order && !isLeftMost) isCurBlock = false;
            nodeIdx = (!isCurBlock)? rightBound
                     :(nodeIdx-1 < leftBound)? nodeIdx
                     :nodeIdx-1;
            
            offset = offset - ((isCurBlock)? 0:step);
            while(alignOrder != 0) {
                if((alignOrder == order && isCurBlock) || forest[offset+2*nodeIdx+1]) {
                    nodeIdx = 2*nodeIdx;
                } else {
                    nodeIdx = 2*nodeIdx+1;
                }
                alignOrder--;
            }
            
            pos[text_idx] = (isCurBlock)? (threadIdx.x-(nodeIdx-blockDim.x)+(forest[offset+nodeIdx]))
                           :(step-nodeIdx+threadIdx.x);
        }
    }
}

void CountPosition(const char *text, int *pos, int text_size)
{
    int threads = 512;
    int nblocks = ((text_size-1)/threads+1);
    int aligned_txt_size = nblocks*threads;
    int step  = 2*threads; 
    int order = (int)log2((float)threads);
    
    bool *binIndexedForest;
    cudaMalloc(&binIndexedForest, sizeof(bool)*aligned_txt_size*2);
    cudaMemset(&binIndexedForest, 0, sizeof(bool)*aligned_txt_size*2);
    
    // build binary indexed trees
    IndexLeafNode<<<nblocks, threads>>>(text, binIndexedForest, text_size, step);
    for(int h = order-1; h >= 0; --h) {
        int base = (int)pow(2, h);
        threads /= 2;
        IndexInteranlNode<<<nblocks, threads>>>(binIndexedForest, base, step);
    }
    
    // traverse
    threads = 512;
    FindPos<<<nblocks, threads>>>(pos, binIndexedForest, text_size, order, step);
    
    cudaFree(binIndexedForest);
}

struct equal_to_one {
	__host__ __device__ bool operator()(const int x) const {
        return (x == 1);
	}
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size);
	cudaMemset(&head, 0, sizeof(int)*text_size);
    thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), cumsum_d(buffer);
    
    // TODO
    thrust::sequence(cumsum_d, cumsum_d+text_size);
    thrust::copy_if(cumsum_d, cumsum_d+text_size, pos_d, head_d, equal_to_one());
    
    int *head_h = (int*)malloc( sizeof(int)*text_size );
    cudaMemcpy(head_h, head, sizeof(int)*text_size, cudaMemcpyDeviceToHost);
    nhead = text_size;
    for(int i = text_size-1; i >= 0; --i) {
        if(head_h[i] != 0) break;
        --nhead;
    }
    
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
