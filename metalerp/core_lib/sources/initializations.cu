
#define INCLUDE_METALERP_INTERNAL_MACRO_UTILS

#include "../include/metalerp.h"

//fundamental
type signBias;
BOOL32 METALERP_CUDAMODE;
BOOL8 METALERP_CUDA_AVAILABLE;



//combos

//base variant params

type min_A_E;
type max_A_E;
type min_A_O;
type max_A_O;

type min_D_E;
type max_D_E;
type min_D_O;
type max_D_O;

//parametric variant params
type p_min_A_E;
type p_min_A_O;

type p_min_D_E;
type p_min_D_O;

#ifndef ODD_EVEN_SAME_HYPERPARAMS
p_StoredParams_Amax_E MAX_PARAMS_AE;
p_StoredParams_Amax_O MAX_PARAMS_AO;
p_StoredParams_Dmax_E MAX_PARAMS_DE;
p_StoredParams_Dmax_O MAX_PARAMS_DO;

#else
p_StoredParams_Amax_E MAX_PARAMS_AE;
p_StoredParams_Dmax_E MAX_PARAMS_DE;
#endif

//combos
enum Functions RightArm;
enum Functions LeftArm;
enum Functions_LR RightArm_LR;
enum Functions_LR LeftArm_LR;
/*************/
//initializations

void base_HP_init()
{
    min_A_E = cast(type, 0);
    max_A_E = cast(type, METALERP_MAXES_INIT);
    min_A_O = cast(type, 0);
    max_A_O = cast(type, METALERP_MAXES_INIT);
    min_D_E = cast(type, 0);
    max_D_E = cast(type, METALERP_MAXES_INIT);
    min_D_O = cast(type, 0);
    max_D_O = cast(type, METALERP_MAXES_INIT);

    COPY_PARAM2DEVICE(D_bParams, min_A_E)
    COPY_PARAM2DEVICE(D_bParams, max_A_E)
    COPY_PARAM2DEVICE(D_bParams, min_A_O)
    COPY_PARAM2DEVICE(D_bParams, max_A_O)
    COPY_PARAM2DEVICE(D_bParams, min_D_E)
    COPY_PARAM2DEVICE(D_bParams, max_D_E)
    COPY_PARAM2DEVICE(D_bParams, min_D_O)
    COPY_PARAM2DEVICE(D_bParams, max_D_O)

}

void advanced_HP_init()
{
    p_min_A_E = cast(type, 0);
    p_min_A_O = cast(type, 0);

    p_min_D_E = cast(type, 0);
    p_min_D_O = cast(type, 0);
    //
    p_max_A_E = cast(type, METALERP_MAXES_INIT);
    p_max_A_O = cast(type, METALERP_MAXES_INIT);

    p_max_D_E = cast(type, METALERP_MAXES_INIT);
    p_max_D_O = cast(type, METALERP_MAXES_INIT);
    //-----------------
    p_Z_ascEven = cast(type, 1);
    p_max_A_E_combine_Z = cast(type, p_Z_ascEven * p_max_A_E);

    p_Z_ascOdd = cast(type, 1);
    p_max_A_O_combine_Z = cast(type, p_Z_ascOdd * p_max_A_O);

    p_Z_descEven = cast(type, 1);
    p_max_D_E_combine_Z = cast(type, p_Z_descEven * p_max_D_E);

    p_Z_descOdd = cast(type, 1);
    p_max_D_O_combine_Z = cast(type, p_Z_descOdd * p_max_D_O);
    //
    p_K_ascEven = cast(type, 1);
    p_max_A_E_combine_K = cast(type, p_K_ascEven * p_max_A_E);

    p_K_ascOdd = cast(type, 1);
    p_max_A_O_combine_K = cast(type, p_K_ascOdd * p_max_A_O);

    p_K_descEven = cast(type, 1);
    p_max_D_E_combine_K = cast(type, p_K_descEven * p_max_D_E);

    p_K_descOdd = cast(type, 1);
    p_max_D_O_combine_K = cast(type, p_K_descOdd * p_max_D_O);


    COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
    COPY_PARAM2DEVICE(D_pParams, p_min_A_O)
    COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_Z)

    COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
    COPY_PARAM2DEVICE(D_pParams, p_min_D_O)
    COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_Z)

    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
    COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_K)
    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
    COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_K)
}

void HyperParams_init()
{
    base_HP_init();
    advanced_HP_init();
}

void combos_init()
{
    RightArm = cast(enum Functions, 0);
    LeftArm = cast(enum Functions, 0);
    RightArm_LR = cast(enum Functions_LR, 0);
    LeftArm_LR = cast(enum Functions_LR, 0);

    COPY_PARAM2DEVICE_ENUM(LeftArm, RightArm)
    COPY_PARAM2DEVICE_ENUM(LeftArm_LR, RightArm_LR)
}

void metalerp_checkCUDA(BOOL8* availabilityVar)
{
    int count = 0;
    cudaError_t errc = cudaGetDeviceCount(&count);
    if(errc != cudaSuccess)
    {
        fprintf(stderr, "warning - metalerp cuda initialization:\nexperienced issues while fetching CUDA-capable devices. CUDA parallel processing won't be available for this runtime.\nError received: %s\n", cudaGetErrorString(errc));
        *availabilityVar = 0;
        return;
    }
    if(count <= 0)
    {
        fprintf(stderr, "no CUDA-capable devices were detected.\n");
        *availabilityVar = 0;
    }
    else 
    {
        *availabilityVar = 1;
        printf("check cuda successful\n");
    }
}

void metalerp_CUDA_init()
{
    signBias = cast(type, 1);

    COPY_PARAM2DEVICE(D_signBias, signBias)
    
    HyperParams_init();
    
    combos_init();

    METALERP_CUDAMODE = 0;
    METALERP_CUDA_AVAILABLE = 0;
    metalerp_checkCUDA(&METALERP_CUDA_AVAILABLE);

    /*
    METALERP_CUDA_AVAILABLE = 1;
    METALERP_CUDAMODE = 1;
    */
}


//interface-level        
void setCUDA_Mode(BOOL32 num)
{
    if(num && METALERP_CUDA_AVAILABLE)
        {
        METALERP_CUDAMODE = 1;
            
        //full lib parameters copy sync routine
        COPY_PARAM2DEVICE(D_signBias, signBias)

        COPY_PARAM2DEVICE(D_bParams, min_A_E)
        COPY_PARAM2DEVICE(D_bParams, max_A_E)
        COPY_PARAM2DEVICE(D_bParams, min_A_O)
        COPY_PARAM2DEVICE(D_bParams, max_A_O)
        COPY_PARAM2DEVICE(D_bParams, min_D_E)
        COPY_PARAM2DEVICE(D_bParams, max_D_E)
        COPY_PARAM2DEVICE(D_bParams, min_D_O)
        COPY_PARAM2DEVICE(D_bParams, max_D_O)

        COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
        COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
        COPY_PARAM2DEVICE(D_pParams, p_min_A_O)
        COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_Z)
        COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
        COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
        COPY_PARAM2DEVICE(D_pParams, p_min_D_O)
        COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_Z)
        COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
        COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_K)
        COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
        COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_K)  
        
        COPY_PARAM2DEVICE_ENUM(LeftArm, RightArm)
        COPY_PARAM2DEVICE_ENUM(LeftArm_LR, RightArm_LR)
        }
    else if(!num)
        METALERP_CUDAMODE = 0;
    else
        fprintf(stderr, "Cannot enable CUDA processing as no devices on this machine possess CUDA capability.\n");
}

#undef INCLUDE_METALERP_INTERNAL_MACRO_UTILS
