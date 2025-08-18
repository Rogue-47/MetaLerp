
#ifndef ML_INITIALIZATIONS
#define ML_INITIALIZATIONS

//can initialize the lib simply with this macro at the very beginning of main
#define METALERP_INIT metalerp_init();

#ifndef __CUDACC__
#include "external/xoshiro256plusplus.h"
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#endif

#include "combos.h"



void metalerp_init(); //NOTE: only function to call in main

#define METALERP_MAXES_INIT 100
/*OS-specific*/
#if _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#define GET_FREQ(freq) QueryPerformanceFrequency((freq))
#define MEASURE(start_or_end) QueryPerformanceCounter((start_or_end))
#define calcElapsed(frequency, start, end) ( ( (double) ( ((end).QuadPart) - ((start).QuadPart) ) ) / ( (double) ((frequency).QuadPart) ) )
#define SLEEP(ms) Sleep((ms))

#else
#include <unistd.h>
#define GET_FREQ(freq)
#define MEASURE(start_or_end) clock_gettime(CLOCK_MONOTONIC, (start_or_end))
#define calcElapsed(frequency, start, end) ((end).tv_nsec) < ((start).tv_nsec) ? \
        (( (double)( ((end).tv_sec) - ((start).tv_sec)) ) - 1.0) + ( ((double) ( ( (((end).tv_nsec) + 1000000000L) - ((start).tv_nsec)) ) / (1e+9)) )   \
    :   ( (double)( ((end).tv_sec) - ((start).tv_sec)) ) + ( ((double)( ((end).tv_nsec) - ((start).tv_nsec) ) ) / (1e+9))

#define SLEEP(ms) usleep(((unsigned int)(ms))*((unsigned int)1000)) /*this takes microsecs so the conversion is necessary*/

#endif

void base_HP_init();

void advanced_HP_init();

void HyperParams_init();

void combos_init();

#ifndef __CUDACC__

#ifdef METALERP_FP_MODE
static const float metalerp_imaxOffsetted = ((float)INT_MAX)+1.f;
STATIC_FORCE_INLINE float metalerpRand()
{
    return ((float)(((int64_t)next())>>32))/(metalerp_imaxOffsetted);
}
#else
STATIC_FORCE_INLINE uint64_t metalerpRand()
{
    return next();
}
#endif


//MP initial runtime heuristics functionality
static const size_t metalerp_startNum = 1<<15;
static const size_t metalerp_maxAlloc = 1<<28; //this is the maximum the heuristics function will allocate before exiting and assigning that as the multi-processing threshold switch

extern size_t MP_threshold;
#define METALERP_AVERAGING_SIZE 5
#define METALERP_VOTES_MAX 5
#define METALERP_THRESHOLD_DIFFERENCE 4096
#define METALERP_ST_BIAS 4096 //single-threaded biasing offset
extern int unsigned MP_dispatch_chunksize;

BOOL32 metalerp_determineFaster(type* restrict in, type* restrict out, size_t length); //1 if single-threaded is faster, 0 if MP is faster

STATIC_FORCE_INLINE type
metalerp_add(type x, type y)
{
    return x + y;
}
STATIC_FORCE_INLINE type
metalerp_sub(type x, type y)
{
    return x - y;
}
STATIC_FORCE_INLINE type
metalerp_mul(type x, type y)
{
    return x * y;
}
STATIC_FORCE_INLINE type
metalerp_fma(type x, type y)
{
    return type_fma(x, y, y);
}

typedef type (*metalerpArithOp) (type, type);
static const metalerpArithOp metalerp_heuristicsOps[] = {metalerp_add, metalerp_sub, metalerp_mul, metalerp_fma};
#define METALERP_ARRSIZE(arr) (sizeof((arr)) / (sizeof((arr)[0])))
void metalerp_MP_heuristics();
void metalerp_OMP_init();
#endif

void metalerp_checkCUDA(BOOL8* availabilityVar);
void metalerp_CUDA_init();
void setCUDA_Mode(BOOL32 num);


#endif //initheader