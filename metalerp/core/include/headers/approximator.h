#ifndef APPROXIMATOR_FUNCTIONS_H
#define APPROXIMATOR_FUNCTIONS_H
/*
- definitions of current approximator functions
- their host and device dispatchers
- 
*/

#include "kDispatcher.h"


#ifdef __CUDACC__

/*variables*/

/*setter declarations and definitions*/

    #define METALERP_INTERNAL_APPROXIMATOR(funcName) METALERP_INTERNAL_KERNEL(funcName)
    
    #define METALERP_DEFINE_APPROXIMATOR(funcName) 

    #define METALERP_DEFINE_SETTER(fullFuncSignature, definition) fullFuncSignature{definition}

#elif defined(METALERP_CUDA_LAYER_READY)

/*variables*/

/*setter declarations*/

    #define METALERP_INTERNAL_APPROXIMATOR(funcName) STATIC_FORCE_INLINE type NM(funcName) 

    #define METALERP_DEFINE_APPROXIMATOR(funcName) \
        void deviceDispatch_##funcName(const type* __restrict__ in, type* __restrict out, size_t len); \
        \
        METALERP_DEF_DISPATCHER(hostDispatch_##funcName)    \
        {   \
            \
            METALERP_LOOP_HEADER    \
            {   \
                out[i] = funcName(in[i]);   \
            }   \
        }   \
            \
        METALERP_DEF_DISPATCHER(MPDispatch_##funcName)  \
        {   \
            METALERP_MACRO_PRAGMA(omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads()))   \
            METALERP_LOOP_HEADER    \
            {   \
                out[i] = funcName(in[i]);   \
            }   \
        }   \
        \
        METALERP_DEF_OUTER_DISPATCHER(batched_##funcName)   \
        {   \
            METALERP_SAFETY_CHECKS(batched_##funcName)  \
            \
            METALERP_CUDA_CHECK \
            {   \
                METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_##funcName);   \
                return;     \
            }   \
            \
            METALERP_MP_CHECK   \
            {   \
                METALERP_DISPATCHER_CALL(hostDispatch_##funcName);  \
            }   \
            else    \
            {       \
                METALERP_DISPATCHER_CALL(MPDispatch_##funcName);  \
            }       \
        }
        #define METALERP_DEFINE_SETTER(fullFuncSignature, definition) fullFuncSignature;
#else
    #define METALERP_INTERNAL_APPROXIMATOR(funcName) STATIC_FORCE_INLINE type NM(funcName) 
#endif /*CUDA-host switching*/

    #define METALERP_DefApproximator(funcName)   \
        METALERP_INTERNAL_DEVFUNC   \
        type NM(funcName)(type x);  \
                                    \
        METALERP_DEFINE_APPROXIMATOR(funcName)  \
                                                \
        METALERP_INTERNAL_APPROXIMATOR(funcName)
/**************/

typedef struct 
{
    type min, max_K, max_V;
} sigmackParams;

typedef struct /*takes std (stDeviation) and mean*/
{
    type min, mean, kParam, max_V, vParam, stDeviation, minFactor, maxFactor, kFactor;
} normDistPDFParams;

/*magic number time (jk, refer to the desmos approximators sheet)*/
#define METALERP_SIGMACK_OFFSET cast(type, 0.5)
#define METALERP_SIGMACK_MIN cast(type, 0)
#define METALERP_SIGMACK_MAX cast(type, 0.5)
#define METALERP_SIGMACK_K cast(type, 3.9)
#define METALERP_SIGMACK_V cast(type, 1.3)

#define METALERP_NORMDIST_MINFACTOR cast(type, -0.13)
#define METALERP_NORMDIST_MAXFACTOR cast(type, 2.5)
#define METALERP_NORMDIST_KFACTOR cast(type, 1.1)
#define METALERP_NORMDIST_VPARAM cast(type, 1)

extern sigmackParams sigmackApproxParams;
extern normDistPDFParams normDistApproxParams;

#ifdef __CUDACC__

    #define METALERP_NM_APPROX(identifier, param, index) D_##identifier[index]
    /*same order as in the struct*/
    __device__ type D_sigmackApproxParams[3]; 

    __device__ type D_normDistApproxParams[4];

    #define SIGMACKCOPY_PARAMS2DEVICE() \
    if(METALERP_CUDAMODE) \
    {   \
        CUDA_CALL(cudaMemcpyToSymbol(D_sigmackApproxParams, &(sigmackApproxParams.min), sizeof(type), 0, cudaMemcpyHostToDevice)) \
        CUDA_CALL(cudaMemcpyToSymbol(D_sigmackApproxParams, &(sigmackApproxParams.max_K), sizeof(type), sizeof(type), cudaMemcpyHostToDevice))  \
        CUDA_CALL(cudaMemcpyToSymbol(D_sigmackApproxParams, &(sigmackApproxParams.max_V), sizeof(type), sizeof(type)*2, cudaMemcpyHostToDevice)) \
    }

    #define NORMDISTCOPY_PARAMS2DEVICE()    \
    if(METALERP_CUDAMODE) \
    {   \
        CUDA_CALL(cudaMemcpyToSymbol(D_normDistApproxParams, &(normDistApproxParams.min), sizeof(type), 0, cudaMemcpyHostToDevice)) \
        CUDA_CALL(cudaMemcpyToSymbol(D_normDistApproxParams, &(normDistApproxParams.mean), sizeof(type), sizeof(type), cudaMemcpyHostToDevice))  \
        CUDA_CALL(cudaMemcpyToSymbol(D_normDistApproxParams, &(normDistApproxParams.kParam), sizeof(type), sizeof(type)*2, cudaMemcpyHostToDevice)) \
        CUDA_CALL(cudaMemcpyToSymbol(D_normDistApproxParams, &(normDistApproxParams.max_V), sizeof(type), sizeof(type)*3, cudaMemcpyHostToDevice)) \
    }
#else
    #define SIGMACKCOPY_PARAMS2DEVICE()
    #define NORMDISTCOPY_PARAMS2DEVICE()
    #define METALERP_NM_APPROX(identifier, param, index) identifier.param
#endif

/* setter interface*/

/*v is the z parameter in other approximators*/
METALERP_DEFINE_SETTER
(   /*0.5 offset not included as tweakable since it seems reasonable to leave it as constant for the function's best tunable approximation set*/
void setSigmackParams(type min, type max, type k, type v),
type funcMax = SET_MAX(max);
sigmackApproxParams.min = SET_MIN(min, funcMax);
sigmackApproxParams.max_K = k * funcMax;
sigmackApproxParams.max_V = v * funcMax; 
SIGMACKCOPY_PARAMS2DEVICE()
)

METALERP_DEFINE_SETTER
(   
void resetSigmackParams(),
sigmackApproxParams.min = METALERP_SIGMACK_MIN;
sigmackApproxParams.max_K = METALERP_SIGMACK_MAX * METALERP_SIGMACK_K;
sigmackApproxParams.max_V = METALERP_SIGMACK_MAX * METALERP_SIGMACK_V; 
SIGMACKCOPY_PARAMS2DEVICE()
)


METALERP_DEFINE_SETTER
(   /*0.5 offset not included as tweakable since it seems reasonable to leave it as constant for the function's best tunable approximation set*/
void setNormDistParams(type standardDeviation, type mean),

normDistApproxParams.mean = mean;

normDistApproxParams.stDeviation = type_abs(standardDeviation);

type funcMax = 1/(normDistApproxParams.maxFactor * normDistApproxParams.stDeviation);

normDistApproxParams.min = normDistApproxParams.minFactor / normDistApproxParams.stDeviation;

normDistApproxParams.kParam = 
normDistApproxParams.stDeviation / (normDistApproxParams.kFactor * funcMax);

normDistApproxParams.max_V = funcMax * normDistApproxParams.vParam;

NORMDISTCOPY_PARAMS2DEVICE()
)

METALERP_DEFINE_SETTER
(   /*0.5 offset not included as tweakable since it seems reasonable to leave it as constant for the function's best tunable approximation set*/
void setNormDistTunableParams(type vParam, type minFactor, type maxFactor, type kFactor),

normDistApproxParams.vParam = vParam;

normDistApproxParams.minFactor = minFactor;

normDistApproxParams.maxFactor = maxFactor;

normDistApproxParams.kFactor = kFactor;

setNormDistParams(normDistApproxParams.stDeviation, normDistApproxParams.mean);
)

METALERP_DEFINE_SETTER
(   /*0.5 offset not included as tweakable since it seems reasonable to leave it as constant for the function's best tunable approximation set*/
void resetNormDistParams(),

setNormDistTunableParams(METALERP_NORMDIST_VPARAM, METALERP_NORMDIST_MINFACTOR, METALERP_NORMDIST_MAXFACTOR, METALERP_NORMDIST_KFACTOR);
setNormDistParams(1, 0); //default std=1 and mean=0
)

/*the kernels*/

METALERP_INTERNAL_DEVFUNC 
type NM(sig_RightArm)(type x)
{
    type t = x/(x + METALERP_NM_APPROX(sigmackApproxParams, max_K, 1));
    return type_fma(t, METALERP_NM_APPROX(sigmackApproxParams, max_V, 2), 
    type_fma(-t, METALERP_NM_APPROX(sigmackApproxParams, min, 0), METALERP_SIGMACK_OFFSET) );
}

METALERP_INTERNAL_DEVFUNC 
type NM(sig_LeftArm)(type x)
{
    type t = x/(METALERP_NM_APPROX(sigmackApproxParams, max_K, 1) - x);
    return type_fma(t, METALERP_NM_APPROX(sigmackApproxParams, max_V, 2), 
    type_fma(-t, METALERP_NM_APPROX(sigmackApproxParams, min, 0), METALERP_SIGMACK_OFFSET) );
}

/*fast and accurate Normal Distribution's CDF (sigmoid/standard logistic function) approximation*/
METALERP_DefApproximator(Sigmack)(type x) /*Sigmack = Sig(moid) + (Car)mack*/
{
    return (x >= 0 ? NM(sig_RightArm(x)) : NM(sig_LeftArm(x)));
}

/*fast and accurate Normal Distribution's PDF (Gaussian) approximation*/
METALERP_DefApproximator(NormDistApproximator)(type x)
{
    type xShifted = x - METALERP_NM_APPROX(normDistApproxParams, mean, 1);
    xShifted *= xShifted;
    type t = METALERP_NM_APPROX(normDistApproxParams, kParam, 2) / ( METALERP_NM_APPROX(normDistApproxParams, kParam, 2) + xShifted); 
    return type_max(0, 
    LERP(METALERP_NM_APPROX(normDistApproxParams, min, 0), METALERP_NM_APPROX(normDistApproxParams, max_V, 3), t));
}


#endif  //approximators header