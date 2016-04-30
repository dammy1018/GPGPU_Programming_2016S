#include "lab3.h"
#include <cstdio>
#include <cmath>
#include "Timer.h"

#define MAX_LEVEL     4
#define NUM_ITERATION 10000

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void CalculateFixed(
    const float *background, 
    const float *target, 
    const float *mask, 
    float *fixed,
    const int wb, const int hb, const int wt, const int ht, 
    const int oy, const int ox
)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and mask[curt] > 127.0f) {
        bool nt_bnd = (yt == 0), wt_bnd = (xt == 0), st_bnd = (yt == ht-1), et_bnd = (xt == wt-1);
        int North_t = (nt_bnd)? curt:(curt-wt);
        int West_t  = (wt_bnd)? curt:(curt-1);
        int South_t = (st_bnd)? curt:(curt+wt);
        int East_t  = (et_bnd)? curt:(curt+1);
        
        fixed[curt*3+0] = 4.0f*target[curt*3+0]-(target[North_t*3+0]+target[West_t*3+0]+target[South_t*3+0]+target[East_t*3+0]);
        fixed[curt*3+1] = 4.0f*target[curt*3+1]-(target[North_t*3+1]+target[West_t*3+1]+target[South_t*3+1]+target[East_t*3+1]);
        fixed[curt*3+2] = 4.0f*target[curt*3+2]-(target[North_t*3+2]+target[West_t*3+2]+target[South_t*3+2]+target[East_t*3+2]);
        
        const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            bool nb_bnd = (yb == 0), wb_bnd = (xb == 0), sb_bnd = (yb == hb-1), eb_bnd = (xb == wb-1);
            int North_b = (nb_bnd)? (curb):(curb-wb);
            int West_b  = (wb_bnd)? (curb):(curb-1);
            int South_b = (sb_bnd)? (curb):(curb+wb);
            int East_b  = (eb_bnd)? (curb):(curb+1);
            
            bool isMasked_n = (nt_bnd)? true:(mask[North_t] <= 127.0f);
            bool isMasked_w = (wt_bnd)? true:(mask[West_t]  <= 127.0f);
            bool isMasked_s = (st_bnd)? true:(mask[South_t] <= 127.0f);
            bool isMasked_e = (et_bnd)? true:(mask[East_t]  <= 127.0f);
            
            if(isMasked_n) {
                fixed[curt*3+0] += background[North_b*3+0];
                fixed[curt*3+1] += background[North_b*3+1];
                fixed[curt*3+2] += background[North_b*3+2];    
            } 
            
            if(isMasked_w) {
                fixed[curt*3+0] += background[West_b*3+0];
                fixed[curt*3+1] += background[West_b*3+1];
                fixed[curt*3+2] += background[West_b*3+2];    
            }
            
            if(isMasked_s) {
                fixed[curt*3+0] += background[South_b*3+0];
                fixed[curt*3+1] += background[South_b*3+1];
                fixed[curt*3+2] += background[South_b*3+2];    
            } 

            if(isMasked_e) {
                fixed[curt*3+0] += background[East_b*3+0];
                fixed[curt*3+1] += background[East_b*3+1];
                fixed[curt*3+2] += background[East_b*3+2];    
            } 
		}
	}
}

__global__ void PoissonImageCloningIteration(
    const float *fixed, 
    const float *mask, 
    const float *buf1, 
    float *buf2, 
    const int wt, const int ht
)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
        bool nt_bnd = (yt == 0), wt_bnd = (xt == 0), st_bnd = (yt == ht-1), et_bnd = (xt == wt-1);
        int North_t = (nt_bnd)? curt:(curt-wt);
        int West_t  = (wt_bnd)? curt:(curt-1);
        int South_t = (st_bnd)? curt:(curt+wt);
        int East_t  = (et_bnd)? curt:(curt+1);
            
        bool isMasked_n = (nt_bnd)? true:(mask[North_t] <= 127.0f);
        bool isMasked_w = (wt_bnd)? true:(mask[West_t]  <= 127.0f);
        bool isMasked_s = (st_bnd)? true:(mask[South_t] <= 127.0f);
        bool isMasked_e = (et_bnd)? true:(mask[East_t]  <= 127.0f);
            
        buf2[curt*3+0] = fixed[curt*3+0];
        buf2[curt*3+1] = fixed[curt*3+1];
		buf2[curt*3+2] = fixed[curt*3+2];
            
        if(!isMasked_n) {
            buf2[curt*3+0] += buf1[North_t*3+0];
            buf2[curt*3+1] += buf1[North_t*3+1];
            buf2[curt*3+2] += buf1[North_t*3+2];    
        }
            
        if(!isMasked_w) {
            buf2[curt*3+0] += buf1[West_t*3+0];
            buf2[curt*3+1] += buf1[West_t*3+1];
            buf2[curt*3+2] += buf1[West_t*3+2];    
        }
            
        if(!isMasked_s) {
            buf2[curt*3+0] += buf1[South_t*3+0];
            buf2[curt*3+1] += buf1[South_t*3+1];
            buf2[curt*3+2] += buf1[South_t*3+2];    
        }

        if(!isMasked_e) {
            buf2[curt*3+0] += buf1[East_t*3+0];
            buf2[curt*3+1] += buf1[East_t*3+1];
            buf2[curt*3+2] += buf1[East_t*3+2];    
        }
            
        buf2[curt*3+0] *= 0.25f;
        buf2[curt*3+1] *= 0.25f;
		buf2[curt*3+2] *= 0.25f;
	}
}

__global__ void Shrink_DownSampling(
    float *target,
    const float *source,
    const int wt, const int ht,
    const int ws, const int hs
)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = y*wt+x;
    const int curs = (y*2)*ws+x*2;
    if(y < ht and x < wt) {
        target[curt*3+0] = (source[curs*3+0]+source[(curs+1)*3+0]+source[(curs+ws)*3+0]+source[(curs+ws+1)*3+0])/4.0f;
        target[curt*3+1] = (source[curs*3+1]+source[(curs+1)*3+1]+source[(curs+ws)*3+1]+source[(curs+ws+1)*3+1])/4.0f;
        target[curt*3+2] = (source[curs*3+2]+source[(curs+1)*3+2]+source[(curs+ws)*3+2]+source[(curs+ws+1)*3+2])/4.0f; 
    }
}

__global__ void NN_DownSampling(
    float *target,
    const float *source,
    const int wt, const int ht,
    const int ws, const int hs
)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = y*wt+x;
    const int curs = (y*2)*ws+x*2;
    if(y < ht and x < wt) {
        target[curt*3+0] = source[curs*3+0];
        target[curt*3+1] = source[curs*3+1];
        target[curt*3+2] = source[curs*3+2]; 
    }
}

__global__ void NN_UpSampling(
    float *target,
    const float *source,
    const int wt, const int ht
)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = y*wt+x;
    const int curs = (y/2)*(wt/2)+x/2;
    if(y < ht and x < wt) {
        target[curt*3+0] = source[curs*3+0];
        target[curt*3+1] = source[curs*3+1];
        target[curt*3+2] = source[curs*3+2];    
    }
}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = target[curt*3+0];
            output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

void HierarchicalTraining(
    const float *background,
    const float *target,
    const float *mask,
    float *result,
    const int wb, const int hb, const int wt, const int ht,
    const int oy, const int ox,
    const int level
)
{   
    int nIteration = (level == MAX_LEVEL)? (NUM_ITERATION/pow(2, level)):(NUM_ITERATION/pow(2, level)/10);
    dim3 gdim_f(CeilDiv(wt,32), CeilDiv(ht,16)), bdim_f(32,16);
    dim3 gdim_c(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), bdim_c(32,16);
    dim3 gdim_b(CeilDiv(wb/2,32), CeilDiv(hb/2,16)), bdim_b(32,16); 
    
    float *background_c;
    float *target_c;
    float *mask_c;
    float *coarse_result;
    float *fixed, *buf1, *buf2;
    
    cudaMalloc(&fixed, sizeof(float)*wt*ht*3);
    cudaMalloc(&buf1, sizeof(float)*wt*ht*3);
    cudaMalloc(&buf2, sizeof(float)*wt*ht*3);
    
    if(level != MAX_LEVEL) {
        cudaMalloc(&background_c, sizeof(float)*(wb/2)*(hb/2)*3);
        cudaMalloc(&target_c, sizeof(float)*(wt/2)*(ht/2)*3);
        cudaMalloc(&mask_c, sizeof(float)*(wt/2)*(ht/2)*3);
        cudaMalloc(&coarse_result, sizeof(float)*(wt/2)*(ht/2)*3);
        
        Shrink_DownSampling<<<gdim_b, bdim_b>>>(background_c, background, wb/2, hb/2, wb, hb);
        Shrink_DownSampling<<<gdim_c, bdim_c>>>(target_c, target, wt/2, ht/2, wt, ht);
        NN_DownSampling<<<gdim_c, bdim_c>>>(mask_c, mask, wt/2, ht/2, wt, ht);
        
        HierarchicalTraining(
            background_c, target_c, mask_c, coarse_result, 
            wb/2, hb/2, wt/2, ht/2, oy/2, ox/2, (level+1)
        );
        
        NN_UpSampling<<<gdim_f, bdim_f>>>(buf1, coarse_result, wt, ht);
    } else {
        cudaMemcpy(buf1, target, wt*ht*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }
    
    CalculateFixed<<<gdim_f, bdim_f>>>(
        background, target, mask, fixed,
        wb, hb, wt, ht, oy, ox
    );
    
    for(int i = 0; i < nIteration; ++i) {
        PoissonImageCloningIteration<<<gdim_f, bdim_f>>>(fixed, mask, buf1, buf2, wt, ht);
        PoissonImageCloningIteration<<<gdim_f, bdim_f>>>(fixed, mask, buf2, buf1, wt, ht);
    }
    
    cudaMemcpy(result, buf1, wt*ht*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    
    if(level != MAX_LEVEL) {
        cudaFree(background_c);
        cudaFree(target_c);
        cudaFree(mask_c);
        cudaFree(coarse_result);
    }
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
    Timer timer_count_position;
    timer_count_position.Start();
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    
    float *result;
    cudaMalloc(&result, sizeof(float)*wt*ht*3);
    
    HierarchicalTraining(
        background, target, mask, result,
        wb, hb, wt, ht, oy, ox, 0
    );
    
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(
        background, result, mask, output,
        wb, hb, wt, ht, oy, ox
    );
    
    timer_count_position.Pause();
	printf_timer(timer_count_position);
}
