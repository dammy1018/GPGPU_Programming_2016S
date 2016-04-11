#include "lab2.h"
#include <stdlib.h>
#include <iostream> 
#include <time.h>   
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
using namespace std;

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
static const unsigned NITERATION = 30;
static const float TIME_STEP = 1.f/24.f;
//static const float SPACE_STEP   = 1.f;
static const float SWIRL_RADIUS = 10.f;

// ====================================================================================================================
// Stable Flued Implementation
// ====================================================================================================================
// ~~~ CUDA Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__device__ bool checkBoundary(int blockIdx, int blockDim, int threadIdx){
    int x = threadIdx;
    int y = blockIdx;
    return (x == 0 || x == (blockDim-1) || y == 0 || y == 479);
}

__global__ void mInitVelocity(float *u_dimX, float *u_dimY) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    u_dimX[Idx] = 0.f;
    u_dimY[Idx] = 0.8f/(float)(blockIdx.x+1);
}

__global__ void mInitDensity(float *d_curr) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)threadIdx.x;
    float y = (float)blockIdx.x;
    float length = sqrt(((x-100.0)*(x-100.0))+((y-100.0)*(y-100.0)));
    
    if(length < 20) {
        d_curr[Idx] = 255; 
    }
}

__global__ void mInitForce(float *f_dimX, float *f_dimY) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)threadIdx.x;
    float y = (float)blockIdx.x;
    float length = sqrt((float)((x-320)*(x-320))+(float)((y-240)*(y-240)));
    
    if(length < SWIRL_RADIUS) {
        f_dimX[Idx] = (240.0-y)/length;    
        f_dimY[Idx] = (x-320.0)/length;
    } else {
        f_dimX[Idx] = f_dimY[Idx] = 0.f;
    }
}

__global__ void mAddDensity(float *dense, float *dense_old, float dt) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    dense[Idx] += dense_old[Idx]*dt;
}

__global__ void mAddForce_TwoDim(float *velocityX, float *velocityY, float *forceX, float *forceY, float dt) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    velocityX[Idx] = (velocityX[Idx] >= 0.6)? velocityX[Idx]:velocityX[Idx]+forceX[Idx]*dt;
    velocityY[Idx] = (velocityY[Idx] >= 0.6)? velocityY[Idx]:velocityY[Idx]+forceY[Idx]*dt;
}

__global__ void mAddExternForce(float *w_dimX, float *w_dimY, float *f_dimX, float *f_dimY, float dt) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    w_dimX[Idx] = -0.5*w_dimX[Idx];
    w_dimY[Idx] = -0.5*w_dimY[Idx]; 
}

__global__ void mAdvect(float *new_data, float *old_data, float *xv, float *yv, float t_step, float s_stepX, float s_stepY) {
    if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    float curr_x = (float)threadIdx.x;
    float curr_y = (float)blockIdx.x;
    float last_x = curr_x - t_step*s_stepX*xv[Idx];
    float last_y = curr_y - t_step*s_stepY*yv[Idx];
    
    if(last_x < 1.5)   last_x = 1.5;
    if(last_x > 637.5) last_x = 637.5;
    if(last_y < 1.5)   last_y = 1.5;
    if(last_y > 477.5) last_y = 477.5;
    
    // Bilinear Interpolation
    float xDiff = last_x - (int)last_x;
    float yDiff = last_y - (int)last_y;
    int LeftTopX = (int)last_x;
    int LeftTopY = (int)last_y;
    int LeftTopIdx = LeftTopY * blockDim.x + LeftTopX;
    new_data[Idx] = (xDiff*yDiff)*old_data[LeftTopIdx+blockDim.x+1]
                  +(xDiff*(1.f-yDiff))*old_data[LeftTopIdx+1]
                  +((1.f-xDiff)*yDiff)*old_data[LeftTopIdx+blockDim.x]
                  +((1.f-xDiff)*(1.f-yDiff))*old_data[LeftTopIdx];
}

__global__ void mJocobi_TwoDim(float *x_new, float *x_old, float* b, float alpha, float rBeta) {
    if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Left   = Idx - 1;
    int Right  = Idx + 1;
    int Top    = Idx + blockDim.x;
    int Bottom = Idx - blockDim.x;
    
    x_new[Idx] = ((x_old[Left]+x_old[Right]+x_old[Top]+x_old[Bottom])*alpha + b[Idx])*rBeta;
}

__global__ void mDivergence_TwoDim(float *div, float *u_dimX, float *u_dimY, float r_sStep) {
    if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Left   = Idx - 1;
    int Right  = Idx + 1;
    int Top    = Idx + blockDim.x;
    int Bottom = Idx - blockDim.x;
    
    div[Idx] = ((u_dimX[Right]-u_dimX[Left])+(u_dimY[Top]-u_dimY[Bottom]))*r_sStep;
}

__global__ void mGradient_TwoDim(float *u_dimX, float *u_dimY, float *scalar, float coeffX, float coeffY) {
    if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) return;
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Left   = Idx - 1;
    int Right  = Idx + 1;
    int Top    = Idx + blockDim.x;
    int Bottom = Idx - blockDim.x;
    
    u_dimX[Idx] -= (scalar[Right] - scalar[Left])*coeffX;
    u_dimY[Idx] -= (scalar[Top] - scalar[Bottom])*coeffY;
}


__global__ void mSetFieldBoundary(float *field, float scalar) {
    if(checkBoundary(blockIdx.x, blockDim.x, threadIdx.x)) {
        int Idx = blockIdx.x * blockDim.x + threadIdx.x;
        int x = threadIdx.x;
        int y = blockIdx.x;
        
        if(x == 0 && y == 0) {
            field[Idx] = field[Idx+blockDim.x+1]*scalar;
        } else if(x == 0 && y == blockDim.x-1) {
            field[Idx] = field[Idx-blockDim.x+1]*scalar;
        } else if (x == blockDim.x-1 && y == 0) {
            field[Idx] = field[Idx+blockDim.x-1]*scalar;
        } else if (x == blockDim.x-1 && y == blockDim.x-1) {
            field[Idx] = field[Idx-blockDim.x-1]*scalar;
        } else if (x == 0) {
            field[Idx] = field[Idx+1]*scalar;
        } else if(x == blockDim.x-1) {
            field[Idx] = field[Idx-1]*scalar;
        } else if(y == 0) {
            field[Idx] = field[Idx+blockDim.x]*scalar;
        } else field[Idx] = field[Idx-blockDim.x]*scalar;
    } else return;
}

__global__ void mAttachTexture(uint8_t *frame, float *dense) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    frame[Idx] = (dense[Idx] > 255.0)? 255:(uint8_t)(dense[Idx]);
}


__global__ void mAddDrip(float *dense, int centerX, int centerY, float redius) {
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x;
    float length = sqrt((float)((x-centerX)*(x-centerX))+(float)((y-centerY)*(y-centerY)));
    
    if(length < redius) {
        dense[Idx] += 200;   
    }
}

// ~~~ Stable Fluid Host Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template<class T>
void Swap(T *scr1, T *src2) {
    T *tmp;
    cudaMalloc(&tmp, sizeof(T)*W*H);
    cudaMemcpy(tmp, scr1, sizeof(T)*W*H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(scr1, src2, sizeof(T)*W*H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(src2, tmp, sizeof(T)*W*H, cudaMemcpyDeviceToDevice);
    cudaFree(tmp);
}

void Diffuse(float *dict, float *src, float coeff, float dt, float scalar) {
    float alpha = dt*coeff*(float)W*(float)H;
    float rbata = 1.f/(1.f+4.f*alpha);
    
    float *next_dict;
    cudaMalloc(&next_dict, sizeof(float)*W*H);
    cudaMemcpy(next_dict, dict, sizeof(float)*W*H, cudaMemcpyDeviceToDevice);
    
    for(unsigned i = 0; i < NITERATION; ++i) {
        mJocobi_TwoDim<<<H, W>>>(next_dict, dict, src, alpha, rbata);
        mSetFieldBoundary<<<H, W>>>(next_dict, scalar);
        Swap(dict, next_dict);
    }
    
    cudaFree(next_dict);
}

// ====================================================================================================================
// Lab2VideoGenerator Class
// ====================================================================================================================
struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
    srand((unsigned)time(NULL));
}

Lab2VideoGenerator::~Lab2VideoGenerator() {
    if(d_curr) cudaFree(d_curr);
    if(d_last) cudaFree(d_last);
    if(u_dimX) cudaFree(u_dimX);
    if(u_dimY) cudaFree(u_dimY);
    if(w_dimX) cudaFree(w_dimX);
    if(w_dimY) cudaFree(w_dimY);
    if(f_dimX) cudaFree(f_dimX);
    if(f_dimY) cudaFree(f_dimY);
    if(p)      cudaFree(p);
}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
}

void Lab2VideoGenerator::init() {
    cudaMalloc(&d_curr, sizeof(float)*W*H);
    cudaMalloc(&d_last, sizeof(float)*W*H);
    cudaMalloc(&u_dimX, sizeof(float)*W*H);
    cudaMalloc(&u_dimY, sizeof(float)*W*H);
    cudaMalloc(&w_dimX, sizeof(float)*W*H);
    cudaMalloc(&w_dimY, sizeof(float)*W*H);
    cudaMalloc(&f_dimX, sizeof(float)*W*H);
    cudaMalloc(&f_dimY, sizeof(float)*W*H);
    cudaMalloc(&p, sizeof(float)*W*H);

    // Initial Texture
    thrust::device_ptr<float> dc_ptr(d_curr), dl_ptr(d_last);
    thrust::fill(dc_ptr, dc_ptr+W*H, 0.f);
    thrust::fill(dl_ptr, dl_ptr+W*H, 0.f);
    
    thrust::device_ptr<float> ux_ptr(u_dimX), uy_ptr(u_dimY), wx_ptr(w_dimX), wy_ptr(w_dimY), p_ptr(p);
    thrust::fill(ux_ptr, ux_ptr+W*H, 0.f);
    thrust::fill(uy_ptr, uy_ptr+W*H, 0.f);
    thrust::fill(wx_ptr, wx_ptr+W*H, 0.f);
    thrust::fill(wy_ptr, wy_ptr+W*H, 0.f);
    thrust::fill(p_ptr, p_ptr+W*H, 0.f);
    
    mInitVelocity<<<H, W>>>(u_dimX, u_dimY);
    mInitDensity<<<H, W>>>(d_curr);
    mInitForce<<<H, W>>>(f_dimX, f_dimY);
}

void Lab2VideoGenerator::UpdateAccelerateField() {
    mAddForce_TwoDim<<<H, W>>>(u_dimX, u_dimY, w_dimX, w_dimX, TIME_STEP);
    Swap(w_dimX, u_dimX);
    Swap(w_dimY, u_dimY);
    
    Diffuse(u_dimX, w_dimX, 0.00002, TIME_STEP, -1.f);
    Diffuse(u_dimY, w_dimY, 0.00002, TIME_STEP, -1.f);
    
    UpdatePressureField();
    UpdateVelocityField();
    Swap(w_dimX, u_dimX);
    Swap(w_dimY, u_dimY);
    
    mAdvect<<<H, W>>>(u_dimX, w_dimX, w_dimX, w_dimY, TIME_STEP, (float)W, (float)H);
    mAdvect<<<H, W>>>(u_dimY, w_dimY, w_dimX, w_dimY, TIME_STEP, (float)W, (float)H);
    mSetFieldBoundary<<<H, W>>>(u_dimX, -1.f);
    mSetFieldBoundary<<<H, W>>>(u_dimY, -1.f);

    UpdatePressureField();
    UpdateVelocityField();
}

void Lab2VideoGenerator::UpdatePressureField() {
    float alpha = 1.0;
    float rbata = 1.f/4.f;
    float *div;
    float *next_p;
    
    cudaMalloc(&div, sizeof(float)*W*H);
    cudaMalloc(&next_p, sizeof(float)*W*H);
    thrust::device_ptr<float> p_ptr(p);
    thrust::fill(p_ptr, p_ptr+W*H, 0.f);
    cudaMemcpy(next_p, p, sizeof(float)*W*H, cudaMemcpyDeviceToDevice);
    
    mDivergence_TwoDim<<<H, W>>>(div, u_dimX, u_dimY, -0.5/(float)W);
    
    for(unsigned i = 0; i < NITERATION; ++i) {
        mJocobi_TwoDim<<<H, W>>>(next_p, p, div, alpha, rbata);
        mSetFieldBoundary<<<H, W>>>(next_p, 1.0);
        Swap(p, next_p);
    }
    
    cudaFree(div);
    cudaFree(next_p);
}

void Lab2VideoGenerator::UpdateVelocityField() {
    mGradient_TwoDim<<<H, W>>>(u_dimX, u_dimY, p, 0.5*(float)W, 0.5*(float)H);
    mSetFieldBoundary<<<H, W>>>(u_dimX, -1.0);
    mSetFieldBoundary<<<H, W>>>(u_dimY, -1.0);
}

void Lab2VideoGenerator::UpdateDensityField(bool isDrip, int cX, int cY, int r) {
    Swap(d_last, d_curr);
    Diffuse(d_curr, d_last, 0.00008, TIME_STEP, 1.0);
    Swap(d_last, d_curr);
    mAdvect<<<H, W>>>(d_curr, d_last, u_dimX, u_dimY, TIME_STEP, (float)W, (float)H);
    if(isDrip) mAddDrip<<<H, W>>>(d_curr, cX, cY, r);
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
    if((impl->t) == 0) {
        init();
    } else {
        bool DripWater = (rand()%4 == 0);
        int radius  = rand()%15+12;
        int centerX = (rand()%(W-32)+15);
        int centerY = (rand()%(H-32)+15);
        UpdateAccelerateField();
        UpdateDensityField(DripWater, centerX, centerY, radius);
        //if((impl->t)%80 == 0) mAddExternForce<<<H, W>>>(u_dimX, u_dimY, f_dimX, f_dimY, TIME_STEP);
        Swap(u_dimX, w_dimX);
        Swap(u_dimY, w_dimY);
    }
    
    mAttachTexture<<<H, W>>>(yuv, d_curr);
    
    //cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}

