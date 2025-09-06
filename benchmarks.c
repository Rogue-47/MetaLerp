
//full benchmarking for the lib's variants' speeds (mean single-execution speed in a batched and single-call routine)
//full OS platform, build flags used, and hardware disclosure is available at the top of the benchmarks output text file.
//although the lib's CUDA side is fully functional - it would not yield great results to test
//the lib's device routines on my MX350 GPU as it only has 2GB VRAM and cannot take enough data past the point
//where GPU-processing starts to really beat both single and multi-processed, well-vectorized SIMD execution on
//a decent micrprocessor speed-wise. So for now I could only provide host-side processing results generated locally
//via the C/C++ interface (fastest setting for the lib).

#define METALERP_FAST
#define METALERP_CUDA_LAYER_READY
#define NO_INVERSION_WARNINGS

#include "metalerp/auxiliary/metalerpTest.h"


/*

#compile with

nvcc -DMETALERP_FAST -g -O3 -use_fast_math -Xptxas="-O3" \
 -Xcompiler="-Ofast -ffast-math -fno-math-errno -mfma -funroll-loops -falign-functions=64 -fprefetch-loop-arrays -march=native -mtune=native -mavx -mavx2 -mf16c -msse4.2" \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_61,code=compute_61 \
  -c metalerp/core/sources/initializations.cu -o CUmetalerp.o && \
nvcc -DMETALERP_FAST -g -O3 -use_fast_math -Xptxas="-O3" \
 -Xcompiler="-Ofast -ffast-math -fno-math-errno -mfma -funroll-loops -falign-functions=64 -fprefetch-loop-arrays -march=native -mtune=native -mavx -mavx2 -mf16c -msse4.2" \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_61,code=compute_61 \
  -c metalerp/auxiliary/sources/benchmarks.cu -o BMmetalerp.o && \
gcc -g -DMETALERP_FAST -Ofast -ffast-math -fno-math-errno -mfma -funroll-loops -falign-functions=64 -fprefetch-loop-arrays -march=native -mtune=native -mavx -mavx2 -mf16c -msse4.2 -flto -fopenmp \
  -o b benchmarks.c metalerp/core/sources/initializations.c metalerp/core/sources/externals/externals_init.c metalerp/auxiliary/sources/test_utils.c CUmetalerp.o BMmetalerp.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lm


*/

int main()
{

    METALERP_INIT

    benchmarks_init();

    setNormDistParams(0.5f, 1); //to match the benchmarks' normal distribution.
    
    static const size_t ARRSIZE = cast(size_t, 1)<<28; //with float native type specification, this is exactly 1GiB of data

    printf("\n--------------Direct Comparisons between functions (Sigmack and NormDistApproximator are the lib's approximators):\n--------------\n");
    
    ALLOCATE_IN_OUT(ARRSIZE)

    BATCHED_VS_Measurement(ARRSIZE, batched_Sigmack, batched_Sigmoid);
    
    averagePerfDynamicInput(ARRSIZE, Sigmack)
    averagePerfDynamicInput(ARRSIZE, Sigmoid)

    BATCHED_VS_Measurement(ARRSIZE, batched_NormDistApproximator, batched_NormalDistributionPDF);

    averagePerfDynamicInput(ARRSIZE, NormDistApproximator)
    averagePerfDynamicInput(ARRSIZE, NormalDistributionPDF)

    BATCHED_VS_Measurement(ARRSIZE, batched_Sigmack, batched_fastSig);

    averagePerfDynamicInput(ARRSIZE, Sigmack)
    averagePerfDynamicInput(ARRSIZE, fastSig)

    BATCHED_VS_Measurement(ARRSIZE, batched_Sigmack, batched_fastSig2);

    averagePerfDynamicInput(ARRSIZE, Sigmack)
    averagePerfDynamicInput(ARRSIZE, fastSig2)

    //

    BATCHED_VS_Measurement(ARRSIZE, batched_NormDistApproximator, batched_ReLU);

    averagePerfDynamicInput(ARRSIZE, NormDistApproximator)
    averagePerfDynamicInput(ARRSIZE, ReLU)

    BATCHED_VS_Measurement(ARRSIZE, batched_NormDistApproximator, batched_LeakyReLU);

    averagePerfDynamicInput(ARRSIZE, NormDistApproximator)
    averagePerfDynamicInput(ARRSIZE, LeakyReLU)

    BATCHED_VS_Measurement(ARRSIZE, batched_Sigmack, batched_ReLU);

    averagePerfDynamicInput(ARRSIZE, Sigmack)
    averagePerfDynamicInput(ARRSIZE, ReLU)

    BATCHED_VS_Measurement(ARRSIZE, batched_Sigmack, batched_LeakyReLU);

    averagePerfDynamicInput(ARRSIZE, Sigmack)
    averagePerfDynamicInput(ARRSIZE, LeakyReLU)

    FREE_IN_OUT(ARRSIZE)

    printf("*****************\nMetaLerp and Benchmarks (BM) Performance Speed Measurements (Batched Measurements)\n*****************\n");

    printf("base: \n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_B_A_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_B_A_O)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_B_D_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_B_D_O)

    printf("base inverses: \n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_B_A_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_B_A_O)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_B_D_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_B_D_O)

    printf("parametric: \n");

    VEC_averagePerfDynamicInput(ARRSIZE, batched_P_A_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_P_A_O)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_P_D_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_P_D_O)

    printf("parametric inverses: \n");

    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_P_A_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_P_A_O)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_P_D_E)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_inv_P_D_O)

    printf("hybrids (with random arms set): \n");

    int Left = next() % METALERP_HYBRID_ARM_TABLE_SIZE;
    int Right = next() % METALERP_HYBRID_ARM_TABLE_SIZE;
    int Left_LR = next() % METALERP_HYBRID_LR_ARM_TABLE_SIZE;
    int Right_LR = next() % METALERP_HYBRID_LR_ARM_TABLE_SIZE;
    setHybridComboArms(Left, Right);
    setHybridComboArms_LR(Left_LR, Right_LR);

    VEC_averagePerfDynamicInput(ARRSIZE, batched_Hybrid)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Hybrid_LR)

    printf("Approximators: \n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Sigmack)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_NormDistApproximator)

    printf("Comparative baseline transformations: \n");

    printf("classic 2D lerp: (min and max constants: 1, 10 respectively)\n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_LERP)

    printf("Sigmoid and some of its popular approximations: \n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Sigmoid)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_fastSig)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_fastSig2)

    printf("Classic Machine Learning functions: \n");

    VEC_averagePerfDynamicInput(ARRSIZE, batched_ReLU)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_LeakyReLU)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_softPlus)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Mish)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Swish)

    printf("trig and hyperbolic trig:\n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Cos)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Sin)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Tan)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Tan_h)

    printf("log functions (natural log and Log10):\n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Ln)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_BM_Log10)
    
    printf("Very basic arithmetic kernels: \n");
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Increment)
    VEC_averagePerfDynamicInput(ARRSIZE, batched_Decrement)

    return 0;

}

