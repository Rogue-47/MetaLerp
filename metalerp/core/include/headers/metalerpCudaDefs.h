#ifndef METALERP_CU_DEFS
#define METALERP_CU_DEFS
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
#endif //CENTRALIZED CUDA BACKEND HEADER