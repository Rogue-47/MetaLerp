/*
This lib internally does its computations with the statically defined types in this file
this is much faster than employing a truly dynamic and run-time flexible data type-agnostic system
that gives you the choice (like numpy) and is necessary for this lib's main goal of being very light and fast;
due to this - its internal configuration is only changeable through compilation of the source code with specific macro definitions

e.g.: basically, if really high precision (double floating-point that is) is needed, you'll have to recompile
with the flag -DTYPE_FLOAT64 in the compilation command
multiple makefile targets will be offered for big configurations of the lib (speed, security, native dtype)
*/

//core macros undefined by the interface header 
//(due to their likely probability of name-clashing with other workspace environment definitions and declarations)
//should be handled separately outside of the header's include guards here
//this ensures other headers that need to individually include the common header
//(most likely headers part of the official lib, included individually, or the external headers reliant on this core header like benchmarks.h or perfMeasurement)
//the interface (metalerp.h) header is included and does its undefs can still operate with its definitions separately

#ifndef cast
#define cast(Type, val) ((Type)val)
#endif
//
#ifndef MINIMUM
#define MINIMUM METALERP_MINIMUM
#endif
//
#ifndef MAXIMUM
    #ifdef MLMAX32
    #define MAXIMUM FLT_MAX
    #define type_abs fabsf              
    #define type_fma fmaf
    #define type_min fminf
    #define type_max fmaxf
    #elif defined(MLMAX64) 
    #define MAXIMUM DBL_MAX
    #define type_abs fabs
    #define type_fma fma
    #define type_min fmin
    #define type_max fmax
    #elif defined(MLMAX16)
    #define MAXIMUM __FLT16_MAX__
    #define type_abs(x) fabsf(((float)(x)))              
    #define type_fma(x, y, z) fmaf(((float)(x)), ((float)(y)), ((float)(z))) 
    #define type_min(x, y) fminf(((float)(x)), ((float)(y)))
    #define type_max(x, y) fmaxf(((float)(x)), ((float)(y)))
    #elif defined(MLMAX_I32)
    #define MAXIMUM INT_MAX
    #define type_abs abs
    #define type_fma(x, y, z) (((x) * (y)) + (z))
    #define type_min(a, b) ((a) < (b)) ? a : b
    #define type_max(a, b) ((a) > (b)) ? a : b
    #elif defined(MLMAX_I64)
    #define MAXIMUM LLONG_MAX
    #define type_abs llabs
    #define type_fma(x, y, z) (((x) * (y)) + (z))
    #define type_min(a, b) ((a) < (b)) ? a : b
    #define type_max(a, b) ((a) > (b)) ? a : b
    #endif
#endif
//

#ifndef type
#define type METALERP_NATIVE_COMPUTATION_TYPE
#endif

#ifndef NM
    
    #ifndef __CUDACC__
    #define NM(param) param
    #else
    #define NM(param) D_##param
    #endif

#endif


#ifndef COMMONS_H /*main header*/
#define COMMONS_H


#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

typedef unsigned char BOOL8;
typedef int32_t BOOL32;

    #if !defined(TYPE_INT32) && !defined(TYPE_INT64)
        #define METALERP_FP_MODE
    #else
        #define INT_MODE
    #endif //determine mode

    #if !defined(TYPE_FLOAT64) && !defined(TYPE_FLOAT16) && defined(METALERP_FP_MODE)
    typedef float METALERP_NATIVE_COMPUTATION_TYPE;
    
    /* //maybe un-comment in the future if the lib ever expands to the point where building for targets with undefined standard limits becomes a goal
    typedef union 
    {
        int i;
        float f;
    } type_limit;
    const type_limit TYPE_MAX = {.i = 0x7F7FFFFF}; //IEEE compliant finite float32 upper limit
    const type_limit TYPE_FRAC_MIN = {.i = 0x00800000}; //smallest 32-bit normalized positive fraction
    #define MAXIMUM TYPE_MAX.f
    #define MINIMUM TYPE_FRAC_MIN.f
    */
    static const type METALERP_MINIMUM = FLT_EPSILON * 2;
    #define MAXIMUM FLT_MAX
    #define MLMAX32
    #define type_abs fabsf              
    #define type_fma fmaf
    #define type_min fminf
    #define type_max fmaxf
    
    #elif defined(TYPE_FLOAT16)
    typedef _Float16 METALERP_NATIVE_COMPUTATION_TYPE;
    #define MAXIMUM __FLT16_MAX__
    #define MLMAX16
    static const type METALERP_MINIMUM = FLT16_EPSILON * 2;
    #define type_abs(x) fabsf(((float)(x)))              
    #define type_fma(x, y, z) fmaf(((float)(x)), ((float)(y)), ((float)(z))) 
    #define type_min(x, y) fminf(((float)(x)), ((float)(y)))
    #define type_max(x, y) fmaxf(((float)(x)), ((float)(y)))
    #ifdef DBG
        #define upcastF16(x)   ((float)(x)) //just specific to debugging and testing
    #endif
    #elif defined(TYPE_FLOAT64)
    typedef double METALERP_NATIVE_COMPUTATION_TYPE;
    /*
     typedef union 
    {
        long long i;
        double f;
    } type_limit;                     
    const type_limit TYPE_MAX = {.i = 0x7FEFFFFFFFFFFFFF}; //IEEE compliant finite float64 upper limit
    const type_limit TYPE_FRAC_MIN = {.i = 0x0010000000000000}; //smallest 64-bit normalized positive fraction
    #define MAXIMUM TYPE_MAX.f
    #define MINIMUM TYPE_FRAC_MIN.f
    */
    #define MAXIMUM DBL_MAX
    #define MLMAX64
    static const type METALERP_MINIMUM = DBL_EPSILON * 2;
    #define type_abs fabs
    #define type_fma fma
    #define type_min fmin
    #define type_max fmax
    /*integers for this library's functions seems like a useless thing,
    but they may prove important to keep for applications other than the common transformation pipelines with heavy floating-point reliance (graphics, dynamics, ML, etc.)
    */
    
    #elif defined(TYPE_INT32)
    typedef int32_t METALERP_NATIVE_COMPUTATION_TYPE;
    #define MAXIMUM INT_MAX
    #define MLMAX_I32
    #define MINIMUM 1
    #define type_abs abs
    #define type_fma(x, y, z) (((x) * (y)) + (z))
    #define type_min(a, b) ((a) < (b)) ? a : b
    #define type_max(a, b) ((a) > (b)) ? a : b
    
    #elif defined(TYPE_INT64)
    typedef int64_t METALERP_NATIVE_COMPUTATION_TYPE;
    #define MAXIMUM LLONG_MAX
    #define MLMAX_I64
    #define MINIMUM (int64_t)1
    #define type_abs llabs
    #define type_fma(x, y, z) (((x) * (y)) + (z))
    #define type_min(a, b) ((a) < (b)) ? a : b
    #define type_max(a, b) ((a) > (b)) ? a : b
    #endif //TYPE INFERENCE


#ifdef METALERP_INTERFACE_LIBMODE
#define STATIC_FORCE_INLINE
#else
#define STATIC_FORCE_INLINE static inline __attribute__((__always_inline__))
#endif

#if !defined(LIB_METALERP)


//probably compiler (GCC) specific, testing with clang hasn't been done yet, and surely msvc would complain about this  

#include "platformDefs.h"
#include <stdio.h>
#include <string.h>

    /*****************/
    //the heterogenuous runtime

    extern BOOL32 METALERP_CUDAMODE; //runtime flag to switch between device and host functionality
    extern BOOL8 METALERP_CUDA_AVAILABLE;
    //setters are referenced from the CUDA object file
    #if __CUDACC__

        #define METALERP_REDEF_LATER

        #if !defined(METALERP_RELEASE) && !defined(METALERP_FAST)
        #include<assert.h>
            #define CUDA_CALL(func_call)  \
                do  \
                {   \
                    cudaError_t errcode = (func_call);  \
                    if(errcode != cudaSuccess)    \
                    {   \
                        fprintf(stderr, "Cuda Function Call \"%s\" failed to execute with CUDA error: \"%s\"\n", #func_call, cudaGetErrorString(errcode)); \
                        assert(errcode == cudaSuccess); /*will always fail, but it's for extra verbosity, pin-pointing where the function that failed was and terminating runtime*/ \
                    }   \
                } while (0);
            
        #elif !defined(METALERP_FAST)
            #define CUDA_CALL(func_call)  \
                do  \
                {   \
                    cudaError_t errcode = (func_call);  \
                    if(errcode != cudaSuccess)    \
                    {   \
                        fprintf(stderr, "Cuda Function Call \"%s\" failed to execute with error: \"%s\"\n", #func_call, cudaGetErrorString(errcode)); \
                    }   \
                } while (0);
        #else   
            #define CUDA_CALL(func_call) func_call;
        #endif
        
        #include <cuda.h>
        #include <cuda_runtime.h>

        #define METALERP_BLOCK_SIZE cast(size_t, 256)
        
        //TODO: perhaps fix denom as 256 METALERP_BLOCK_SIZE later and observe speed gains
        STATIC_FORCE_INLINE
        size_t metalerp_ceildiv(size_t numer, size_t denom) //strictly internal to the cuda object file
        {
            return ((numer - 1) / denom) + 1;
        }
        
        #define computeGridSize(numElements) metalerp_ceildiv(cast(size_t, numElements), METALERP_BLOCK_SIZE)

        /*wrapper around the interface-level setters for linkage issues resolution that come from calling inlined functions from an object file*/
        #define METALERP_DEF_SHIM(function, type1, arg1) STATIC_FORCE_INLINE void shim_##function(type1 arg1); void function(type1 arg1) {shim_##function(arg1);} STATIC_FORCE_INLINE void shim_##function
        #define METALERP_DEF_SHIM_2ARG(function, type1, arg1, type2, arg2) STATIC_FORCE_INLINE void shim_##function(type1 arg1, type2 arg2); void function(type1 arg1, type2 arg2) {shim_##function(arg1, arg2);} STATIC_FORCE_INLINE void shim_##function

        #define METALERP_EXTERNAL_KERNEL __global__ void //for CUDA kernels, they will be defined in kDispatcher.h
        //host dispatchers of the cuda kernel launches will be defined only when this macro is also defined, since they necessitate the existence of a kernel to launch it in the first place

        #define NM(param) D_##param //naming mode, for function signatures, or global variables.

        #define METALERP_INTERNAL_DEVFUNC __device__ STATIC_FORCE_INLINE

        #define METALERP_INTERNAL_KERNEL(funcName) \
            METALERP_EXTERNAL_KERNEL \
            K_##funcName(type* __restrict__ inArray, size_t len)   \
                {   \
                    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
                    if(idx < len) \
                    {   \
                        inArray[idx] = NM(funcName)(inArray[idx]); \
                    } \
                }   \
            void deviceDispatch_##funcName(const type* __restrict__ in, type* __restrict__ out, size_t len)   \
                {   \
                    size_t numBytes = len * sizeof(type);   \
                    \
                    type *D_in = NULL;   \
                    \
                    CUDA_CALL(cudaMalloc((void**)&D_in, numBytes))  \
                    \
                    CUDA_CALL(cudaMemcpy(D_in, in, numBytes, cudaMemcpyHostToDevice))   \
                    \
                    K_##funcName<<<computeGridSize(len), METALERP_BLOCK_SIZE>>>(D_in, len);  \
                    \
                    CUDA_CALL(cudaMemcpy(out, D_in, numBytes, cudaMemcpyDeviceToHost)) \
                    \
                    CUDA_CALL(cudaFree(D_in))   \
                    \
                }   \
            __device__ STATIC_FORCE_INLINE type NM(funcName) 

        
        #define COPY_PARAM2DEVICE(symbolName, param) if(METALERP_CUDAMODE) {CUDA_CALL(cudaMemcpyToSymbol(symbolName, &(param), sizeof(type), sizeof(type)*D_##param##_idx, cudaMemcpyHostToDevice))} //extra copy to device equivalent at the end of each setter
        #define COPY_PARAM2DEVICE_2(symbolName, param, param2) if(METALERP_CUDAMODE) {CUDA_CALL(cudaMemcpyToSymbol(symbolName, &(param), sizeof(type), sizeof(type)*D_##param##_idx, cudaMemcpyHostToDevice)) CUDA_CALL(cudaMemcpyToSymbol(symbolName, &(param2), sizeof(type), sizeof(type)*D_##param2##_idx, cudaMemcpyHostToDevice))} //extra copy to device equivalent at the end of each setter
        #define COPY_PARAM2DEVICE_ENUM(enumParam_L, enumParam_R)    \
        if(METALERP_CUDAMODE) \
        {\
            CUDA_CALL(cudaMemcpyToSymbol(D_##enumParam_L, &(enumParam_L), sizeof(enum Functions), 0, cudaMemcpyHostToDevice)) \
            CUDA_CALL(cudaMemcpyToSymbol(D_##enumParam_R, &(enumParam_R), sizeof(enum Functions), 0, cudaMemcpyHostToDevice))  \
        }

    //linking the two layers in the final compilation
    #elif defined METALERP_CUDA_LAYER_READY /*last*/

        #define NM(param) param
        
        #define METALERP_INTERNAL_DEVFUNC STATIC_FORCE_INLINE

        #define METALERP_INTERNAL_KERNEL(funcName) void deviceDispatch_##funcName(const type* __restrict__ in, type* __restrict out, size_t len); STATIC_FORCE_INLINE type NM(funcName) 

    #else
        #define METALERP_INTERNAL_DEVFUNC STATIC_FORCE_INLINE

        #define NM(param) param
        
        #define METALERP_INTERNAL_KERNEL(funcName) STATIC_FORCE_INLINE type NM(funcName) 

        

    #endif

    /*****************/

    //CONSTANTS
    static const type minShifted = MINIMUM * cast(type, 3);
        //constants that are only applicable in the case that our native data type is fp of any precision 
        #if defined(METALERP_FP_MODE)
            //relative epsilon shifting (nextafterf doesn't work for some values like 10 when trying to get the next float before it)
            #define SUM_EPSILON(x) + cast(type, type_max(MINIMUM, (type_abs(x)*MINIMUM)))
            #define SUB_EPSILON(x) - cast(type, type_max(MINIMUM, (type_abs(x)*MINIMUM)))
            /*
            greater epsilon shift for better min/max separation which avoids bugs that occur with subtracting then adding epsilon somewhere else (like an open interval calculation in a clamp function)
            to a minimum, initial subtraction so a != b then the addition of an equivalent magnitude epsilon causes it to equal b again then the function relying on the clamp (like an inverse func) works as if a==b is allowed 
            */
            #define SUB_EPSILON2(x) - cast(type, type_max(minShifted, (type_abs(x)*minShifted)))  

            /*will be used for SET_MAX -- ensures b (max boundary) never becomes zero after abs operation 
            (b at b<=0 could easily break the formula in many input cases 
            (boundary disobedience, division by zero, etc.))
            SAFE_B_SETTERS essentially fixes the issues of the function 
            (that are user input-related when setting your b value for any of the forms)
            by making sure b > 0 ALWAYS. this ensures safe behavior of the function even with
            technical user errors, this option makes the function a tiny bit slower when setting
            the hyperparameter b though (which is fine as long as hyperparameter b is not set
            in the tight training loop every iteration)
            NOTE: if negative max behavior needs to be emulated - the library provides functions
            that specialize with this requirement safely while obeying boundaries.
            */
        #else
            #define SUM_EPSILON + cast(type, 1)
            #define SUB_EPSILON - cast(type, 1)
        #endif

        #define OPEN_INTERVAL_MIN(minValue) ((minValue) SUM_EPSILON((minValue)))
        #define OPEN_INTERVAL_MAX(maxValue) ((maxValue) SUB_EPSILON((maxValue)))

    //COMPARISON BIAS (what will the arm-combining functions (mostly just the odd variants and the hybrid arm-comboer) act like at input=0 (ABSOLUTE ZERO))
        #ifndef NEGATIVE_HYBRID_BIAS //negative bias for the hybrid function's partitioning check
            #define HYBRID_POS_COMPARISON >=
        #else
            #define HYBRID_POS_COMPARISON >
        #endif
        #ifndef NEGATIVE_ODD_ASC_BIAS //negative bias for the ascending odd function's check
            #define ASC_POS_COMPARISON >=
        #else
            #define ASC_POS_COMPARISON >
        #endif
        #ifndef NEGATIVE_ODD_DESC_BIAS //negative bias for the descending odd function's check
            #define DESC_POS_COMPARISON >=
        #else
            #define DESC_POS_COMPARISON >
        #endif 

    METALERP_INTERNAL_DEVFUNC
    type NM(branchingSign)(type x);
    METALERP_INTERNAL_DEVFUNC
    type NM(nonBranchingSign)(type x);
    

    void setSignBias(type num);
    //signBias to determine what the sign function yields when it is passed zero (1 for positive 1 as the sign factor, -1 for -1), this is just useful for the branchingSign, the branchless sign function does not rely on this
    
    extern type signBias;
    
    #ifdef __CUDACC__
    __device__ type D_signBias;
    #define D_signBias_idx 0

    METALERP_DEF_SHIM(setSignBias, type, num)(type num)
    {
        signBias = num;
        if( (signBias != 1) && (signBias != -1))
            signBias = 1;
        COPY_PARAM2DEVICE(D_signBias, signBias)
    }

    #endif

    METALERP_INTERNAL_DEVFUNC
    type NM(branchingSign)(type x) //branching sign func
    {
    if(x == ((type)0)) return NM(signBias); //default return for when x equals zero but could be adjusted for biasing the sign factor (at compile time)
    return (x > ((type)0))? (type)1 : (type)-1;
    }     


    #ifdef MLMAX32
    typedef union
    {
        int32_t i; float f;
    } metalerp_intfloat;
    #define METALERP_SGN_RSHIFT_AMOUNT 31
    #define BITWISE_TYPE int32_t
    #elif defined(MLMAX64)
    typedef union
    {
        int64_t i; double f;
    } metalerp_intfloat;
    #define METALERP_SGN_RSHIFT_AMOUNT 63
    #define BITWISE_TYPE int64_t
    #elif defined(MLMAX16)
    typedef union
    {
        int16_t i; _Float16 f;
    } metalerp_intfloat;
    #define BITWISE_TYPE int16_t
    #define METALERP_SGN_RSHIFT_AMOUNT 15
    #elif defined(MLMAX_I32)
    METALERP_SGN_RSHIFT_AMOUNT 31
    #define BITWISE_TYPE int32_t
    #elif defined(MLMAX_I64)
    METALERP_SGN_RSHIFT_AMOUNT 63
    #define BITWISE_TYPE int64_t
    #endif

    METALERP_INTERNAL_DEVFUNC
    type NM(nonBranchingSign)(type x) //less branching (only the sign bias check) but heavily arithmetic
    {
        #ifdef METALERP_FP_MODE
        metalerp_intfloat mix = {.f = (float)x};
        BITWISE_TYPE sgnex = ((mix.i)>>cast(BITWISE_TYPE, METALERP_SGN_RSHIFT_AMOUNT)) & cast(BITWISE_TYPE, 1);
        #else
        BITWISE_TYPE sgnex = (x >> cast(BITWISE_TYPE, METALERP_SGN_RSHIFT_AMOUNT)) & cast(BITWISE_TYPE, 1);
        #endif
        return NM(signBias) == 1 ? (type)(((sgnex)) * (-1) + (!sgnex)) : (type)(!(sgnex) - (sgnex) - 2*(!mix.i));
    }


    //BASIC_OPTIMIZATIONS

        #ifdef NO_HYPERPARAM_CLAMP //less branching for hyperparam setters (adv) -> you have to know what you are doing, k and z
            /*
            #define clamp_paramG(g, newG)
            #define g_Parameter g
            */

            #define clamp_paramK(k, newK)
            #define k_Parameter k

            #define clamp_paramZ(z, newZ)
            #define z_Parameter z

        #else
            /*
            #define clamp_paramG(g, newG) type newG = clampG(g);
            #define g_Parameter c_G
            */

            #define clamp_paramK(k, newK) type newK = clampK(k);
            #define k_Parameter c_K

            #define clamp_paramZ(z, newZ) type newZ = clampZ(z);
            #define z_Parameter c_Z
        #endif

    //SGN macro is the default sign function for unspecified (though with further testing might be removed)
        #if defined(BRANCHLESS_SGN_FUNC)
            #define SGN NBSGN 
            #define BSGN NM(branchingSign)
            #define NBSGN NM(nonBranchingSign)
        #else
            #define SGN BSGN 
            #define BSGN NM(branchingSign)
            #define NBSGN NM(nonBranchingSign)
        #endif

    //LERP OPTIMIZATION, fused multiply-add, maximum of 2 of those for both base and parametric forms
    #define LERP(a, b, t) \
            type_fma((t), (b), type_fma(-(t), (a), (a))) //b passed here should be the normal max except for parametric ascending variants where we'll be passing combined zmax in here, in the descending variant where we're passing already the combined zmax and kmax in the comp macro before LERP

    #define minusMinLERP(a, b, t) \
            type_fma((t), (b), type_fma(-(t), (a), (-a)))

    //BRANCH PREDICTION OPTIMIZATIONS
        #ifndef NO_DOMINANT_INPUT_SIGN //define this if you're totally unsure about your inputs' distribution around zero

            #ifndef MOSTLY_NEGATIVE_INPUTS
            //make use of the fact that typically neural network weights OR dynamic system inputs majorly are positive
                #define EXPECT_BRANCH(condition) __builtin_expect(condition, 1) 
            #else
            //define MOSTLY_NEGATIVE_INPUTS if your case dictates that most of your inputs to these kernels are actually negative
                #define EXPECT_BRANCH(condition) __builtin_expect(condition, 0)
            #endif

        #else
            #define EXPECT_BRANCH(condition) (condition)
        #endif
    //SAFETY for max and min
    /*important note: if you're sure your max (b) will always be positive either because it is always set by some user-defined clamping function
    in the interface layer or because it's always ever set by you or just not automatically set according to some loose rules, for example as if it were
    being used as adjusted learnable or morphing and constantly updating parameters in a loop, you can ensure better speed by defining the following macros
    NOTE: it is important to know that this is here in the first place because
    (1) the formulas break when max <= 0, divisions by zero and such could easily occur when your input x = 0
    (2) for min: min >= max actually works for most forward functions but it breaks most of their inverse variants, especially the descender's inverses... refer to the desmos link*/
        #if defined(METALERP_FP_MODE)
            #ifndef UNSAFE_INVERSE_BOUNDS
                #define clampZMax(zmax, min)    \
                            if(zmax == min) zmax += MINIMUM;
            #else
                #define clampZMax(zmax, min)
            #endif

            #ifdef FAST_MAX_SETTERS
                #define SET_MAX(val) val
                #define ENFORCE_MAX(min, max)
            #else //thought of taking abs(val) here, making the value always the positive magnitude even for those that want a negative max (and only minShifted when 0 is input) since a negative max effect can be achieved with z = -1 using an equivalent parametric variant
                #define SET_MAX(val) val <= 0 ? minShifted : (val);
                #define ENFORCE_MAX(min, max) if((min) >= (max)) (min) = (max) SUB_EPSILON2(max)
            #endif

            #ifdef FAST_MIN_SETTERS
                #define SET_MIN(val, max) val
            #else
                #define SET_MIN(val, max) ((val) < (max)) ? (val) : (max) SUB_EPSILON2(max) 
            #endif
        #endif
    volatile static const char* METALERP_STR = "float was chosen";
    extern BOOL32 metalerpEvenInversionCheckOnce;
    extern BOOL32 metalerpOddInversionCheckOnce;
    //INVERSION QUIRKS - the lib's evenly symmetric variants aren't perfectly invertible (always yield |x|)
    //but the odd variants are perfectly invertible as long as the minimum (a) > 0
    //but they often map 2 inputs to a single output when a <= 0, breaking the horizontal line test
    //(minimum at 0 still causes ambiguity issues for the inverses mathematically but the lib handles it with biasing (sticking to one path (arm) deterministically))
    //so checks will be implemented for now to warn when an inverse is called through the hybrid dispatcher or batched function interface
    //in the case that the minimum < 0.
    //the odd inverse variants have the potential to be safe otherwise via domain restriction for the arms (avoids clashing input having to viable outputs (vice-versa in the case of the odd function), but limits the inverse severely.)
    //or via implementing a piecewise odd inverse that skips the overlapping area between the 2 inverse arms and clamps to the input range of the larger (infinite) non-overlapping areas
    //or reverts to domain restriction WHEN the input to the inverse function is in the region where the inverse arms overlap

    //hybrid dispatch inversion macro will be a bit tedious
        #ifndef NO_INVERSION_WARNINGS
            #define METALERP_CHECK_EVEN_INVERSION(outer_func, fwd_func)  \
            if(!metalerpEvenInversionCheckOnce) { metalerpEvenInversionCheckOnce=1; printf("\n\033[4;31m*************************\n--METALERP WARNING--\n*************************\n\033[0m\nit seems you have called the evenly symmetric function [%s]'s inverse function: [inv_%s] in dispatcher: [%s]\nPlease be aware that the even variants aren't mathematically perfectly invertible, similar to x^2 function\n", #fwd_func, #fwd_func, #outer_func);}

            #define METALERP_CHECK_ODD_INVERSION(outer_func, fwd_func, min) \
            if(((min) <= 0) && !metalerpOddInversionCheckOnce) { metalerpOddInversionCheckOnce=1; printf("\n\033[4;31m*************************\n--METALERP WARNING--\n*************************\n\033[0m\nit seems you have called the oddly symmetric function [%s]'s inverse function: [inv_%s] in dispatcher: [%s] while its minimum was less than zero (%.20f)\nPlease be aware that the odd inversion variants of this lib can produce ambiguous results on some inputs when min < 0 due to domain restriction and arm-biasing around x=0 to ensure (one-one) inverse functionality\n", #fwd_func, #fwd_func, #outer_func, (double)min);}
        #else
            #define METALERP_CHECK_EVEN_INVERSION(outer_func, fwd_func)
            #define METALERP_CHECK_ODD_INVERSION(outer_func, fwd_func, min)
        #endif
#endif //common header development mode inclusion
#endif //metalerp core
