#ifndef COMBOS_H
#define COMBOS_H

#define INSIDE_COMBOS_H

#include "baseForms.h"
#include "advancedForms.h"

#undef INSIDE_COMBOS_H

enum Functions
{
    //base funcs
    B_ASC_EVEN = 0,
    B_ASC_ODD,
    B_DESC_EVEN,
    B_DESC_ODD,
    //parametric equivalents
    P_ASC_EVEN,
    P_ASC_ODD,
    P_DESC_EVEN,
    P_DESC_ODD,
    //inverses
    B_INV_ASC_EVEN,
    B_INV_ASC_ODD,
    B_INV_DESC_EVEN,
    B_INV_DESC_ODD,

    P_INV_ASC_EVEN,
    P_INV_ASC_ODD,
    P_INV_DESC_EVEN,
    P_INV_DESC_ODD,
};

extern enum Functions RightArm;
extern enum Functions LeftArm; 
#ifdef __CUDACC__
__device__ int D_RightArm;
__device__ int D_LeftArm; 
#endif

enum Functions_LR
{
    LR_B_ASC_EVEN = 0,
    LR_B_ASC_ODD_L,
    LR_B_ASC_ODD_R,
    
    LR_B_DESC_EVEN,
    LR_B_DESC_ODD_L,
    LR_B_DESC_ODD_R,

    LR_P_ASC_EVEN,
    LR_P_ASC_ODD_L,
    LR_P_ASC_ODD_R,

    LR_P_DESC_EVEN,
    LR_P_DESC_ODD_L,
    LR_P_DESC_ODD_R,

    LR_B_INV_ASC_EVEN,
    LR_B_INV_ASC_ODD_L,
    LR_B_INV_ASC_ODD_R,

    LR_B_INV_DESC_EVEN,
    LR_B_INV_DESC_ODD_L,
    LR_B_INV_DESC_ODD_R,

    LR_P_INV_ASC_EVEN,
    LR_P_INV_ASC_ODD_L,
    LR_P_INV_ASC_ODD_R,

    LR_P_INV_DESC_EVEN,
    LR_P_INV_DESC_ODD_L,
    LR_P_INV_DESC_ODD_R,
};

extern enum Functions_LR RightArm_LR;
extern enum Functions_LR LeftArm_LR;


#ifdef __CUDACC__

__device__ int D_RightArm_LR;
__device__ int D_LeftArm_LR;

#endif

#ifndef METALERP_INCLUDE_COMBOS_FUNCTIONALITY
typedef type (*scalarVariant) (type);


/*******************/
//Hybrid versions of the odd functions' arms, slightly slower when used in the functions that use both to branch, because they queue more instructions in the branches
//but they're correct for the hybrid arm dispatcher since it's needs more independent arm dispatch to stay simple in overall design

METALERP_INTERNAL_DEVFUNC type
NM(inv_ascendingV_O_RightArm_H)(type y)
{
    type yClamped = NM(clampY_A_R)(y);
    //printf("yClamped retrieved(RA): %f\n", yClamped);
    return NBSGN(min_A_O) * (type_fma(max_A_O, yClamped, -(max_A_O * min_A_O)) / (max_A_O - yClamped));
}

METALERP_INTERNAL_DEVFUNC type
NM(inv_ascendingV_O_LeftArm_H)(type y)
{
    type yClamped = NM(clampY_A_L)(y);
    //printf("yClamped retrieved(LA): %f\n", yClamped);
    return NBSGN(min_A_O) * (type_fma(max_A_O, yClamped, max_A_O * min_A_O) / (max_A_O + yClamped));
}

//parametric
METALERP_INTERNAL_DEVFUNC type
NM(p_inv_ascendingV_O_RightArm_H)(type y)
{
    type yClamped = NM(p_clampY_A_R)(y);
  //  printf("\n-------------------\nPAO-yclamped(R): %.20f\n-------------------\n", yClamped);
    return NBSGN(min_A_O) * (type_fma(p_max_A_O_combine_K, yClamped, 
        -(p_max_A_O_combine_K * p_min_A_O)) 
        / (p_max_A_O_combine_Z - yClamped) );
}
METALERP_INTERNAL_DEVFUNC type
NM(p_inv_ascendingV_O_LeftArm_H)(type y)
{
    type yClamped = NM(p_clampY_A_L)(y);
   /// printf("\n-------------------\nPAO-yclamped(L): %.20f\n-------------------\n", yClamped);
    return NBSGN(min_A_O) * (type_fma(p_max_A_O_combine_K, yClamped, 
        (p_max_A_O_combine_K * p_min_A_O)) 
        / (p_max_A_O_combine_Z + yClamped) );
}

METALERP_INTERNAL_DEVFUNC type
NM(p_inv_descendingV_O_RightArm_H)(type y)
{
    type yClamped = NM(p_clampY_D_R)(y);
  //printf("\n-------------------\nPDO-yclamped(R): %.20f\n-------------------\n", yClamped);
    return NBSGN(p_max_D_O_combine_Z) * ( type_fma(p_max_D_O_combine_K, p_max_D_O_combine_Z,
        -(p_max_D_O_combine_K * yClamped)) 
        / (yClamped - p_min_D_O) );
}
//(kbzb + kby) / (y + a)
METALERP_INTERNAL_DEVFUNC type
NM(p_inv_descendingV_O_LeftArm_H)(type y)
{
    type yClamped = NM(p_clampY_D_L)(y);
 //  printf("\n-------------------\nPDO-yclamped(L): %.20f\n-------------------\n", yClamped);
    return NBSGN(p_max_D_O_combine_Z) * ( type_fma(p_max_D_O_combine_K, p_max_D_O_combine_Z,
        (p_max_D_O_combine_K * yClamped)) 
        / (yClamped + p_min_D_O) );
}
/********************/

#ifndef __CUDACC__
static const scalarVariant allVariants_L[] = /*combo set, parametric and base functions, even and odd variants could be called together now as a single piece-wise around x=0 function*/
{
ascendingVariant_E, ascendingV_O_LeftArm, descendingVariant_E, descendingV_O_LeftArm,
p_ascendingVariant_E, p_ascendingV_O_LeftArm, p_descendingVariant_E, p_descendingV_O_LeftArm,
inv_ascendingVariant_E, inv_ascendingV_O_LeftArm_H, inv_descendingVariant_E, inv_descendingV_O_LeftArm,
p_inv_ascendingVariant_E, p_inv_ascendingV_O_LeftArm_H, p_inv_descendingVariant_E, p_inv_descendingV_O_LeftArm_H
};

static const scalarVariant allVariants_R[] =
{
ascendingVariant_E, ascendingV_O_RightArm, descendingVariant_E, descendingV_O_RightArm,
p_ascendingVariant_E, p_ascendingV_O_RightArm, p_descendingVariant_E, p_descendingV_O_RightArm,
inv_ascendingVariant_E, inv_ascendingV_O_RightArm_H, inv_descendingVariant_E, inv_descendingV_O_RightArm,
p_inv_ascendingVariant_E, p_inv_ascendingV_O_RightArm_H, p_inv_descendingVariant_E, p_inv_descendingV_O_RightArm_H
};

//--------------------------------------------

static const scalarVariant allVariants[] =  /*let the chaos ensue - left arms could be called as right arms and vice-versa,
redundancies could be made more easily, unpredictable behavior AND division by zero could occur if used badly, etc.
********** IMPORTANT NOTE: the desmos link will help a lot regarding visualization when composing the combos. */
{
ascendingVariant_E, ascendingV_O_LeftArm, ascendingV_O_RightArm,
descendingVariant_E, descendingV_O_LeftArm, descendingV_O_RightArm,
p_ascendingVariant_E, p_ascendingV_O_LeftArm, p_ascendingV_O_RightArm,
p_descendingVariant_E, p_descendingV_O_LeftArm, p_descendingV_O_RightArm,
inv_ascendingVariant_E, inv_ascendingV_O_LeftArm_H, inv_ascendingV_O_RightArm_H,
inv_descendingVariant_E, inv_descendingV_O_LeftArm, inv_descendingV_O_RightArm,
p_inv_ascendingVariant_E, p_inv_ascendingV_O_LeftArm_H, p_inv_ascendingV_O_RightArm_H,
p_inv_descendingVariant_E, p_inv_descendingV_O_LeftArm_H, p_inv_descendingV_O_RightArm_H,
};



#else

#define METALERP_FUNCTIONS_COUNT (sizeof(D_allVariants_R) / sizeof(D_allVariants_R[0]))
#define METALERP_FUNCTIONS_LR_COUNT (sizeof(D_allVariants) / sizeof(D_allVariants[0]))


static const __device__ scalarVariant D_allVariants_L[] = /*combo set, parametric and base functions, even and odd variants could be called together now as a single piece-wise around x=0 function*/
{
D_ascendingVariant_E, D_ascendingV_O_LeftArm, D_descendingVariant_E, D_descendingV_O_LeftArm,
D_p_ascendingVariant_E, D_p_ascendingV_O_LeftArm, D_p_descendingVariant_E, D_p_descendingV_O_LeftArm,
D_inv_ascendingVariant_E, D_inv_ascendingV_O_LeftArm_H, D_inv_descendingVariant_E, D_inv_descendingV_O_LeftArm,
D_p_inv_ascendingVariant_E, D_p_inv_ascendingV_O_LeftArm_H, D_p_inv_descendingVariant_E, D_p_inv_descendingV_O_LeftArm_H
};

static const __device__ scalarVariant D_allVariants_R[] =
{
D_ascendingVariant_E, D_ascendingV_O_RightArm, D_descendingVariant_E, D_descendingV_O_RightArm,
D_p_ascendingVariant_E, D_p_ascendingV_O_RightArm, D_p_descendingVariant_E, D_p_descendingV_O_RightArm,
D_inv_ascendingVariant_E, D_inv_ascendingV_O_RightArm_H, D_inv_descendingVariant_E, D_inv_descendingV_O_RightArm,
D_p_inv_ascendingVariant_E, D_p_inv_ascendingV_O_RightArm_H, D_p_inv_descendingVariant_E, D_p_inv_descendingV_O_RightArm_H
};

//--------------------------------------------

static const __device__ scalarVariant D_allVariants[] =  /*let the chaos ensue - left arms could be called as right arms and vice-versa,
redundancies could be made more easily, unpredictable behavior AND division by zero could occur if used badly, etc.
********** IMPORTANT NOTE: the desmos link will help a lot regarding visualization when composing the combos. */
{
D_ascendingVariant_E, D_ascendingV_O_LeftArm, D_ascendingV_O_RightArm,
D_descendingVariant_E, D_descendingV_O_LeftArm, D_descendingV_O_RightArm,
D_p_ascendingVariant_E, D_p_ascendingV_O_LeftArm, D_p_ascendingV_O_RightArm,
D_p_descendingVariant_E, D_p_descendingV_O_LeftArm, D_p_descendingV_O_RightArm,
D_inv_ascendingVariant_E, D_inv_ascendingV_O_LeftArm_H, D_inv_ascendingV_O_RightArm_H,
D_inv_descendingVariant_E, D_inv_descendingV_O_LeftArm, D_inv_descendingV_O_RightArm,
D_p_inv_ascendingVariant_E, D_p_inv_ascendingV_O_LeftArm_H, D_p_inv_ascendingV_O_RightArm_H,
D_p_inv_descendingVariant_E, D_p_inv_descendingV_O_LeftArm_H, D_p_inv_descendingV_O_RightArm_H,
};
#endif


void setHybridComboArms(enum Functions L_Arm, enum Functions R_Arm);

void setHybridComboArms_LR(enum Functions_LR L_Arm, enum Functions_LR R_Arm);

METALERP_INTERNAL_DEVFUNC type NM(hybridVariant)(type x);
METALERP_INTERNAL_DEVFUNC type NM(hybridVariant_LR)(type x);
/*implementations*/
METALERP_INTERNAL_KERNEL(hybridVariant)(type x)
{   
    return (x HYBRID_POS_COMPARISON cast(type, 0)) ?
    NM(allVariants_R)[NM(RightArm)](x) :
    NM(allVariants_L)[NM(LeftArm)](x) ;
}

METALERP_INTERNAL_KERNEL(hybridVariant_LR)(type x)
{
    return (x HYBRID_POS_COMPARISON cast(type, 0)) ?
    NM(allVariants)[NM(RightArm_LR)](x) :
    NM(allVariants)[NM(LeftArm_LR)](x) ;
}

#ifdef __CUDACC__
    static const int metalerp_armsMax = METALERP_FUNCTIONS_COUNT-1;
    STATIC_FORCE_INLINE void clampHybridArms(int *R_arm, int *L_arm)
    {
      *R_arm = (*R_arm > metalerp_armsMax) ? metalerp_armsMax : ((*R_arm < 0) ? 0 : *R_arm);
      *L_arm = (*L_arm > metalerp_armsMax) ? metalerp_armsMax : ((*L_arm < 0) ? 0 : *L_arm);
    }

    static const int metalerp_armsMax_LR = METALERP_FUNCTIONS_LR_COUNT-1;
    STATIC_FORCE_INLINE void clampHybridArms_LR(int *R_arm, int *L_arm)
    {
      *R_arm = (*R_arm > metalerp_armsMax_LR) ? metalerp_armsMax_LR : ((*R_arm < 0) ? 0 : *R_arm);
      *L_arm = (*L_arm > metalerp_armsMax_LR) ? metalerp_armsMax_LR : ((*L_arm < 0) ? 0 : *L_arm);
    }
    
    METALERP_DEF_SHIM_2ARG(setHybridComboArms, enum Functions, L_Arm, enum Functions, R_Arm)(enum Functions L_Arm, enum Functions R_Arm)
    {
        clampHybridArms((int*)&R_Arm, (int*)&L_Arm);
        LeftArm = L_Arm;
        RightArm = R_Arm;
        COPY_PARAM2DEVICE_ENUM(LeftArm, RightArm)
    }

    /***********************************/
    METALERP_DEF_SHIM_2ARG(setHybridComboArms_LR, enum Functions_LR, L_Arm, enum Functions_LR, R_Arm)(enum Functions_LR L_Arm, enum Functions_LR R_Arm)
    {
        clampHybridArms_LR((int*)&R_Arm, (int*)&L_Arm);
        LeftArm_LR = L_Arm;
        RightArm_LR = R_Arm;
        COPY_PARAM2DEVICE_ENUM(LeftArm_LR, RightArm_LR)
    }

#endif //cuda compiled interface funcs

    #ifdef METALERP_REDEF_LATER //baseForms and advancedForms (advancedForms still needs the definitions for same hyperparam mode)

    #undef min_A_E
    #undef max_A_E
    #undef min_A_O
    #undef max_A_O

    #undef min_D_E
    #undef max_D_E
    #undef min_D_O
    #undef max_D_O

    #ifdef ODD_EVEN_SAME_HYPERPARAMS

    #define min_A_E minA
    #define max_A_E maxA
    #define min_A_O minA
    #define max_A_O maxA
    /***********/
    #define min_D_E minD
    #define max_D_E maxD
    #define min_D_O minD
    #define max_D_O maxD

    #else

    #define min_A_E minA_E
    #define max_A_E maxA_E
    #define min_A_O minA_O
    #define max_A_O maxA_O
    /***********/
    #define min_D_E minD_E
    #define max_D_E maxD_E
    #define min_D_O minD_O
    #define max_D_O maxD_O

    #endif //base

    //advancedForms params
        //TODO: currently works only for the default global parameter mode which allows full independence of global parameters (but the issue is fixed in baseForms)

    #undef p_min_A_E 
    #undef p_max_A_E_combine_K 
    #undef p_max_A_E_combine_Z 

    #undef p_min_D_E 
    #undef p_max_D_E_combine_K 
    #undef p_max_D_E_combine_Z 

    #undef p_min_A_O 
    #undef p_max_A_O_combine_K 
    #undef p_max_A_O_combine_Z 

    #undef p_min_D_O 
    #undef p_max_D_O_combine_K 
    #undef p_max_D_O_combine_Z


    #define p_min_A_E p_minA_E
    #define p_max_A_E_combine_K STORED_PARAMS_AE.p_maxA_E_K
    #define p_max_A_E_combine_Z STORED_PARAMS_AE.p_maxA_E_Z

    #define p_min_D_E p_minD_E
    #define p_max_D_E_combine_K STORED_PARAMS_DE.p_maxD_E_K
    #define p_max_D_E_combine_Z STORED_PARAMS_DE.p_maxD_E_Z

    #define p_min_A_O p_minA_O
    #define p_max_A_O_combine_K STORED_PARAMS_AO.p_maxA_O_K
    #define p_max_A_O_combine_Z STORED_PARAMS_AO.p_maxA_O_Z

    #define p_min_D_O p_minD_O
    #define p_max_D_O_combine_K STORED_PARAMS_DO.p_maxD_O_K
    #define p_max_D_O_combine_Z STORED_PARAMS_DO.p_maxD_O_Z

    #endif //redefine the parameter macros back to point to host mirrors


#endif //COMBOS_FUNCTIONALITY

#endif //combos header