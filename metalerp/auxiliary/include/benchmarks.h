#if !defined(BENCHMARKS_H) && !defined(LIB_METALERP)
#define BENCHMARKS_H

#ifdef __cplusplus
extern "C" {
#endif


#ifndef METALERP_FAST
#define METALERP_FAST
#endif

#define INCLUDE_METALERP_INTERNAL_MACRO_UTILS


#define STATIC_FORCE_INLINE static inline __attribute__((__always_inline__))
#define COMMONS_H /*need only the utility macros from the common header*/
#define MLMAX32

#include "../../core/include/headers/commons.h"

#include "../../core/include/headers/metalerpCudaDefs.h"

#include "../../core/include/headers/approximator.h"
#include <stdint.h>
#include <omp.h>
#include <math.h>

#undef COMMONS_H
#undef APPROXIMATOR_FUNCTIONS_H
#undef KERNEL_DISPATCHERS_H

#define COS(X) cosf(X)
#define TANH(X) tanhf(X)
#define LN(X) logf(X)
#define LOG(X) log10f(X)

/*
STATIC_FORCE_INLINE type fastSig(type x);
STATIC_FORCE_INLINE type fastSig2(type x);
STATIC_FORCE_INLINE type ReLU(type x);
STATIC_FORCE_INLINE type LeakyReLU(type x);
STATIC_FORCE_INLINE type Sigmoid(type x);
STATIC_FORCE_INLINE type softPlus(type x);
STATIC_FORCE_INLINE type Mish(type x);
STATIC_FORCE_INLINE type Swish(type x);
STATIC_FORCE_INLINE type ReLUReWritten(type x);
STATIC_FORCE_INLINE type Increment(type x);
STATIC_FORCE_INLINE type Decrement(type x);
*/


typedef int8_t BOOL8;
typedef int32_t BOOL32;
//necessary system globals from the lib
extern BOOL32 METALERP_CUDAMODE;
extern BOOL8 METALERP_CUDA_AVAILABLE;
extern size_t MP_threshold;
extern uint32_t MP_dispatch_chunksize;

#define METALERP_DEF_BENCHMARK METALERP_DEFINE_APPROXIMATOR

void benchmarks_init();

extern type METALERP_NORMDIST_SQRT;
extern type BM_std, BM_mean;
#ifdef __CUDACC__
__device__ type D_METALERP_NORMDIST_SQRT;
__device__ type D_BM_std;
__device__ type D_BM_mean;

#define D_METALERP_NORMDIST_SQRT_idx 0
#define D_BM_std_idx 0
#define D_BM_mean_idx 0
#endif

METALERP_DEF_BENCHMARK(NormalDistributionPDF)(type x)
{   

    type expfDenom = 2 * NM(BM_std) * NM(BM_std);
    type expfNumer = (x - NM(BM_mean));
    expfNumer *= expfNumer;
    return (1/(NM(BM_std) * NM(METALERP_NORMDIST_SQRT))) * expf((-expfNumer) / expfDenom);
}

METALERP_DEF_BENCHMARK(BM_Cos)(type x)
{
    return COS(x);
}
METALERP_DEF_BENCHMARK(BM_Sin)(type x)
{
    return sinf(x);
}
METALERP_DEF_BENCHMARK(BM_Ln)(type x)
{
    return LN(x);
}
METALERP_DEF_BENCHMARK(BM_Log10)(type x)
{
    return LOG(x);
}
METALERP_DEF_BENCHMARK(BM_Tan_h)(type x)
{
    return TANH(x);
}
METALERP_DEF_BENCHMARK(BM_Tan)(type x)
{
    return tanf(x);
}


METALERP_DEF_BENCHMARK(fastSig)(type x)
{
    return x / (cast(type, 1) + type_abs(x));
}

METALERP_DEF_BENCHMARK(fastSig2)(type x)
{
   return (x / (cast(type, 2) * (((x < cast(type, 0)) * (-x)) + ((x >= cast(type, 0)) * x)) + cast(type, 2)) + cast(type, 0.5));
}

METALERP_DEF_BENCHMARK(ReLU)(type x)
{
    return fmaxf(cast(type, 0), x);
}

static const type alpha = 0.3;

METALERP_DEF_BENCHMARK(LeakyReLU)(type x)
{
    return ( x > cast(type, 0) ? x : (alpha * x) );
}

METALERP_DEF_BENCHMARK(Sigmoid)(type x)
{
    return cast(type, 1) / (cast(type, 1) + expf(-x)) ;
}

METALERP_DEF_BENCHMARK(softPlus)(type x)
{
    return logf(cast(type, 1) + expf(x));
}

METALERP_DEF_BENCHMARK(Mish)(type x)
{
    return x * tanhf(NM(softPlus)(x));
}
METALERP_DEF_BENCHMARK(Swish)(type x)
{
    return x / (cast(type, 1) + expf(-x));
}

METALERP_DEF_BENCHMARK(Increment)(type x)
{
    return x+cast(type, 1);
}

METALERP_DEF_BENCHMARK(Decrement)(type x)
{
    return x-cast(type, 1);
}

#define LERP(a, b, t) type_fma((t), (b), type_fma(-(t), (a), (a)))


METALERP_DEF_BENCHMARK(BM_LERP)(type x)
{
    type a = 1, b = 10;
    return LERP(a, b, x);
}




#undef INCLUDE_METALERP_INTERNAL_MACRO_UTILS

#undef BENCHMARKS_H

#ifdef __cplusplus
}
#endif

#endif //BENCHMARKS_H