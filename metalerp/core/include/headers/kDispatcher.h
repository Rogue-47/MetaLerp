/*
math kernel loop wrappers for computing large amounts (batched-computation)
of transformations and storing them, very AVX & SSE-friendly
*/
//kernel dispatchers (CPU functions & CUDA kernels will be wrapped here)
//mid-level (python interface) will interface with these

#ifndef KERNEL_DISPATCHERS_H
#define KERNEL_DISPATCHERS_H

#if !defined(BENCHMARKS_H) 
#include "initializations.h"
#endif //dispatcher macro utilities inclusion

#if !defined(METALERP_RELEASE) && !defined(METALERP_FAST)
        #include <assert.h>
        #define METALERP_DBGMODE

        #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t in_len, const size_t out_len)

        #define METALERP_SAFETY_CHECKS(funcName) \
                assert((in_len == out_len) && in && out && (in_len>0));

        #define METALERP_DEF_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t len)

        #define METALERP_MP_CHECK if(out_len <= MP_threshold)

        #define METALERP_DISPATCHER_CALL(funcName) funcName(in, out, out_len)

        #define METALERP_LOOP_HEADER for(size_t i = 0; i<len; ++i)
        


#elif !defined(METALERP_FAST) //higher tolerance, for production environments

        #define metalerp_min(A, B) (A) < (B) ? (A) : (B)

        #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t in_len, const size_t out_len)

        #define METALERP_SAFETY_CHECKS(funcName) \
            if(!in) {fprintf(stderr, "MetaLerp error at %s call site:\ninput array passed was not initialized properly\n", #funcName);return;}\
            if(!out) {fprintf(stderr, "MetaLerp error at %s call site:\noutput array passed was not initialized properly\n", #funcName);return;}\
            if(in_len == 0) {fprintf(stderr, "MetaLerp error at %s call site:\ninput array length passed was either passed as negative or zero\n", #funcName);return;}\
            if(out_len == 0) {fprintf(stderr, "MetaLerp error at %s call site:\noutput array length passed was either passed as negative or zero\n", #funcName);return;}\
            size_t minlen = metalerp_min(in_len, out_len); /*function will execute even if array sizes are not equal.*/
        
        #define METALERP_DEF_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t minlen)
 
        #define METALERP_MP_CHECK if(minlen <= MP_threshold)

        #define METALERP_DISPATCHER_CALL(funcName) funcName(in, out, minlen)
        
        #define METALERP_LOOP_HEADER for(size_t i = 0; i<minlen; ++i)
        


#else //fastest, used for maximum speed, but also vulnerable if memory/bounds are not checked correctly before the function is called
       
        #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t len)

        #define METALERP_DEF_DISPATCHER(dispatcherName) STATIC_FORCE_INLINE void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t len)
        
        #define METALERP_MP_CHECK if(len <= MP_threshold)

        #define METALERP_DISPATCHER_CALL(funcName) funcName(in, out, len)

        #define METALERP_SAFETY_CHECKS(funcName)
            
        #define METALERP_LOOP_HEADER for(size_t i = 0; i<len; ++i)



#endif


#ifdef METALERP_CUDA_LAYER_READY

    #ifdef METALERP_DBGMODE
        #define ML_LEN out_len

    #elif defined(METALERP_RELEASE)
        #define ML_LEN minlen

    #else

        #define ML_LEN len
    
    #endif

#define METALERP_CUKERNEL_DISPATCHER_CALL(cuFuncName) cuFuncName(in, out, ML_LEN) //dispatcher definer macros are the same for cuda kernel launch dispatchers

#define METALERP_CUDA_CHECK if(METALERP_CUDAMODE)

#define METALERP_DEF_CUDISPATCHER(dispatcherName) void \
                dispatcherName(type* __restrict__ in, type* __restrict__ out, const size_t len)

/*kernel dispatcher definitions*/

#else
#define METALERP_CUDA_CHECK if(0)
#define METALERP_CUKERNEL_DISPATCHER_CALL(cuFuncName)
#endif

#if !defined(BENCHMARKS_H) 

#ifndef __CUDACC__
//declarations
METALERP_DEF_DISPATCHER(hostDispatch_H); //H = Hybrid kernel func
METALERP_DEF_DISPATCHER(MPDispatch_H);
METALERP_DEF_DISPATCHER(hostDispatch_H_LR); //H_LR = left-right arm mixing enabled hybrid kernel func
METALERP_DEF_DISPATCHER(MPDispatch_H_LR);

METALERP_DEF_DISPATCHER(hostDispatch_B_A_E); // B_A_E = base ascending even (evenly symmetric) variant/kernel
METALERP_DEF_DISPATCHER(MPDispatch_B_A_E);
METALERP_DEF_DISPATCHER(hostDispatch_P_A_E); //P = parametric
METALERP_DEF_DISPATCHER(MPDispatch_P_A_E);
METALERP_DEF_DISPATCHER(hostDispatch_B_A_O); //O = odd (symmetry)
METALERP_DEF_DISPATCHER(MPDispatch_B_A_O);
METALERP_DEF_DISPATCHER(hostDispatch_P_A_O);
METALERP_DEF_DISPATCHER(MPDispatch_P_A_O);
METALERP_DEF_DISPATCHER(hostDispatch_B_D_E); //D = descending
METALERP_DEF_DISPATCHER(MPDispatch_B_D_E);
METALERP_DEF_DISPATCHER(hostDispatch_P_D_E);
METALERP_DEF_DISPATCHER(MPDispatch_P_D_E);
METALERP_DEF_DISPATCHER(hostDispatch_B_D_O);
METALERP_DEF_DISPATCHER(MPDispatch_B_D_O);
METALERP_DEF_DISPATCHER(hostDispatch_P_D_O);
METALERP_DEF_DISPATCHER(MPDispatch_P_D_O);

METALERP_DEF_DISPATCHER(hostDispatch_inv_B_A_E);
METALERP_DEF_DISPATCHER(MPDispatch_inv_B_A_E);
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_A_E);
METALERP_DEF_DISPATCHER(MPDispatch_inv_P_A_E);
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_A_O);
METALERP_DEF_DISPATCHER(MPDispatch_inv_B_A_O);
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_A_O);
METALERP_DEF_DISPATCHER(MPDispatch_inv_P_A_O);
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_D_E);
METALERP_DEF_DISPATCHER(MPDispatch_inv_B_D_E);
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_D_E);
METALERP_DEF_DISPATCHER(MPDispatch_inv_P_D_E);
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_D_O);
METALERP_DEF_DISPATCHER(MPDispatch_inv_B_D_O);
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_D_O);
METALERP_DEF_DISPATCHER(MPDispatch_inv_P_D_O);


/******
host interface functions, recommended to call only those unless you're contributing to the library and experimenting with the lower level layers
*******/
//hybrid funcs
METALERP_DEF_OUTER_DISPATCHER(batched_Hybrid)
{
    METALERP_SAFETY_CHECKS(batched_Hybrid)

    METALERP_CUDA_CHECK
    {   
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_hybridVariant);
        return;
    }

    METALERP_MP_CHECK
    { 
        METALERP_DISPATCHER_CALL(hostDispatch_H);
    }
    else
    {   
        METALERP_DISPATCHER_CALL(MPDispatch_H);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_Hybrid_LR)
{
    METALERP_SAFETY_CHECKS(batched_Hybrid_LR)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_hybridVariant_LR);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_H_LR);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_H_LR);
    }    
}

//base
METALERP_DEF_OUTER_DISPATCHER(batched_B_A_E)
{
    METALERP_SAFETY_CHECKS(batched_B_A_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_ascendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_B_A_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_B_A_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_B_A_O)
{
    METALERP_SAFETY_CHECKS(batched_B_A_O)

    METALERP_CUDA_CHECK
    {   
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_ascendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_B_A_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_B_A_O);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_B_D_E)
{
    METALERP_SAFETY_CHECKS(batched_B_D_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_descendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_B_D_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_B_D_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_B_D_O)
{
    METALERP_SAFETY_CHECKS(batched_B_D_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_descendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_B_D_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_B_D_O);
    }    
}
//
METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_A_E)
{
    METALERP_SAFETY_CHECKS(batched_inv_B_A_E)

    METALERP_CHECK_EVEN_INVERSION(batched_inv_B_A_E, inv_ascendingVariant_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_inv_ascendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_B_A_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_B_A_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_A_O)
{
    METALERP_SAFETY_CHECKS(batched_inv_B_A_O)

    METALERP_CHECK_ODD_INVERSION(batched_inv_B_A_O, inv_ascendingVariant_O, min_A_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_inv_ascendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_B_A_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_B_A_O);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_D_E)
{
    METALERP_SAFETY_CHECKS(batched_inv_B_D_E)

    METALERP_CHECK_EVEN_INVERSION(batched_inv_B_D_E, inv_descendingVariant_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_inv_descendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_B_D_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_B_D_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_D_O)
{
    METALERP_SAFETY_CHECKS(batched_inv_B_D_O)

    METALERP_CHECK_ODD_INVERSION(batched_inv_B_D_O, inv_descendingVariant_O, min_D_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_inv_descendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_B_D_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_B_D_O);
    }    
}

//parametric
METALERP_DEF_OUTER_DISPATCHER(batched_P_A_E)
{
    METALERP_SAFETY_CHECKS(batched_P_A_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_ascendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_P_A_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_P_A_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_P_A_O)
{
    METALERP_SAFETY_CHECKS(batched_P_A_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_ascendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_P_A_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_P_A_O);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_P_D_E)
{
    METALERP_SAFETY_CHECKS(batched_P_D_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_descendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_P_D_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_P_D_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_P_D_O)
{
    METALERP_SAFETY_CHECKS(batched_P_D_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_descendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_P_D_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_P_D_O);
    }    
}
//

METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_A_E)
{
    METALERP_SAFETY_CHECKS(batched_inv_P_A_E)

    METALERP_CHECK_EVEN_INVERSION(batched_inv_P_A_E, p_inv_ascendingVariant_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_ascendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_P_A_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_P_A_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_A_O)
{
    METALERP_SAFETY_CHECKS(batched_inv_P_A_O)

    METALERP_CHECK_ODD_INVERSION(batched_inv_P_A_O, p_inv_ascendingVariant_O, p_min_A_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_ascendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_P_A_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_P_A_O);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_D_E)
{
    METALERP_SAFETY_CHECKS(batched_inv_P_D_E)

    METALERP_CHECK_EVEN_INVERSION(batched_inv_P_D_E, p_inv_descendingVariant_E)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_descendingVariant_E);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_P_D_E);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_P_D_E);
    }    
}

METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_D_O)
{
    METALERP_SAFETY_CHECKS(batched_inv_P_D_O)

    METALERP_CHECK_ODD_INVERSION(batched_inv_P_D_O, p_inv_descendingVariant_O, p_min_D_O)

    METALERP_CUDA_CHECK
    {
        METALERP_CUKERNEL_DISPATCHER_CALL(deviceDispatch_p_descendingVariant_O);
        return;
    }

    METALERP_MP_CHECK
    {
        METALERP_DISPATCHER_CALL(hostDispatch_inv_P_D_O);
    }
    else
    {
        METALERP_DISPATCHER_CALL(MPDispatch_inv_P_D_O);
    }    
}
/********************************************************************************************* 
end of batched dispatcher interface*/



//hybrid variants:
METALERP_DEF_DISPATCHER(hostDispatch_H)
{   
    
    METALERP_LOOP_HEADER
    {
        out[i] = hybridVariant(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_H)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = hybridVariant(in[i]);
    }
}

METALERP_DEF_DISPATCHER(hostDispatch_H_LR)
{
    METALERP_LOOP_HEADER
    {
        out[i] = hybridVariant_LR(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_H_LR)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = hybridVariant_LR(in[i]);
    }
}

//ascending
//even
METALERP_DEF_DISPATCHER(hostDispatch_B_A_E) // B_A_E = base ascending even variant/kernel
{
    METALERP_LOOP_HEADER
    {
        out[i] = ascendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_B_A_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = ascendingVariant_E(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_P_A_E) // B_A_E = base ascending even variant/kernel
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_ascendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_P_A_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_ascendingVariant_E(in[i]);
    }
}

//odd

METALERP_DEF_DISPATCHER(hostDispatch_B_A_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = ascendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_B_A_O)
{   
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = ascendingVariant_O(in[i]);
    }
}

//parametric

METALERP_DEF_DISPATCHER(hostDispatch_P_A_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_ascendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_P_A_O)
{   
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_ascendingVariant_O(in[i]);
    }
}

//descending
//even
METALERP_DEF_DISPATCHER(hostDispatch_B_D_E)
{
    METALERP_LOOP_HEADER
    {
        out[i] = descendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_B_D_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = descendingVariant_E(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_P_D_E)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_descendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_P_D_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_descendingVariant_E(in[i]);
    }
}
//odd
METALERP_DEF_DISPATCHER(hostDispatch_B_D_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = descendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_B_D_O)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = descendingVariant_O(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_P_D_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_descendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_P_D_O)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_descendingVariant_O(in[i]);
    }
}

/****** INVERSE ******/
//ascending
//even
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_A_E) // B_A_E = base ascending even variant/kernel
{
    METALERP_LOOP_HEADER
    {
        out[i] = inv_ascendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_B_A_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = inv_ascendingVariant_E(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_A_E) // B_A_E = base ascending even variant/kernel
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_ascendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_P_A_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_ascendingVariant_E(in[i]);
    }
}

//odd

METALERP_DEF_DISPATCHER(hostDispatch_inv_B_A_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = inv_ascendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_B_A_O)
{   
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = inv_ascendingVariant_O(in[i]);
    }
}

//parametric

METALERP_DEF_DISPATCHER(hostDispatch_inv_P_A_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_ascendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_P_A_O)
{   
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_ascendingVariant_O(in[i]);
    }
}

//descending
//even
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_D_E)
{
    METALERP_LOOP_HEADER
    {
        out[i] = inv_descendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_B_D_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = inv_descendingVariant_E(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_D_E)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_descendingVariant_E(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_P_D_E)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_descendingVariant_E(in[i]);
    }
}
//odd
METALERP_DEF_DISPATCHER(hostDispatch_inv_B_D_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = inv_descendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_B_D_O)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = inv_descendingVariant_O(in[i]);
    }
}
//parametric
METALERP_DEF_DISPATCHER(hostDispatch_inv_P_D_O)
{
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_descendingVariant_O(in[i]);
    }
}

METALERP_DEF_DISPATCHER(MPDispatch_inv_P_D_O)
{
    #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
    METALERP_LOOP_HEADER
    {
        out[i] = p_inv_descendingVariant_O(in[i]);
    }
}


#endif //cuda kernel and host dispatcher implementations (inner dispatchers)



#endif //dispatcher macro utilities inclusion


#endif //K_DISPATCHER