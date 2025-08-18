#ifndef ADVANCED_FORMS_H
#define ADVANCED_FORMS_H

#include "commons.h"


/*
defining more advanced versions of the formula
mainly the parameterized forms
arm combiners, whole piecewise behavior combiners
extra variables for each single function,
to emulate functions with double the hyperparameters
gives more modularity in emulating more nuanced 
non-symmetric funcs like Swish or Mish (approximations overload)*/

/* PERF:
- will adopt a pre-computing approach for the hyperparameters k and z, fused b*z and b*k will be stored parameters and used (done)
- resolve and test inefficiencies in the inverse functions (TODO)
*/

//TODO: add CUDA-capability independence later with a macro that enables compiling the source code to only account for microprocessor code



    #ifdef ODD_EVEN_SAME_HYPERPARAMS

        typedef struct
    {
    type p_maxA, p_maxA_K, K_A, p_maxA_Z, Z_A;
    } p_StoredParams_Amax_E;

    typedef struct
    {
    type p_maxD, p_maxD_K, K_D, p_maxD_Z, Z_D;
    } p_StoredParams_Dmax_E;

    extern p_StoredParams_Amax_E STORED_PARAMS_AE; 
    //{cast(type, 1), cast(type, 1), cast(type, 1), cast(type, 1), cast(type, 1)};

    extern p_StoredParams_Dmax_E STORED_PARAMS_DE;
    //{cast(type, 1), cast(type, 1), cast(type, 1), cast(type, 1), cast(type, 1)};

    extern type p_minA;
    extern type p_minD;

    #define p_max_A_E STORED_PARAMS_AE.p_maxA
    #define p_min_A_E p_minA
    #define p_max_D_E STORED_PARAMS_DE.p_maxD
    #define p_min_D_E p_minD

    #define p_max_A_O p_max_A_E
    #define p_min_A_O p_min_A_E
    #define p_max_D_O p_max_D_E
    #define p_min_D_O p_min_D_E
    //tuning hyperparameters (Z, K, G) (_A for ascending variants, _D for descending variants)
    /*
    type G_A = cast(type, 1);
    type G_D = cast(type, 1);

    #define p_G_ascEven G_A
    #define p_G_ascOdd p_G_ascEven

    #define p_G_descEven G_D
    #define p_G_descOdd p_G_descEven
    */

    #define p_Z_ascEven STORED_PARAMS_AE.Z_A
    #define p_Z_ascOdd p_Z_ascEven

    #define p_K_ascEven STORED_PARAMS_AE.K_A
    #define p_K_ascOdd p_K_ascEven

    #define p_Z_descEven STORED_PARAMS_DE.Z_D
    #define p_Z_descOdd p_Z_descEven


    #define p_K_descEven STORED_PARAMS_DE.K_D
    #define p_K_descOdd p_K_descEven

    #define p_max_D_E_combine_K STORED_PARAMS_DE.p_maxD_K
    #define p_max_D_O_combine_K p_max_D_E_combine_K

    #define p_max_A_E_combine_K STORED_PARAMS_AE.p_maxA_K
    #define p_max_A_O_combine_K p_max_A_E_combine_K

    #define p_max_D_E_combine_Z STORED_PARAMS_DE.p_maxD_Z
    #define p_max_D_O_combine_Z p_max_D_E_combine_Z

    #define p_max_A_E_combine_Z STORED_PARAMS_AE.p_maxA_Z
    #define p_max_A_O_combine_Z p_max_A_E_combine_Z

    #define MAX_PARAMS_AE STORED_PARAMS_AE
    #define MAX_PARAMS_AO STORED_PARAMS_AE
    #define MAX_PARAMS_DE STORED_PARAMS_DE
    #define MAX_PARAMS_DO STORED_PARAMS_DE

    #define p_StoredParams_Amax_O p_StoredParams_Amax_E
    #define p_StoredParams_Dmax_O p_StoredParams_Dmax_E

    #else //independent hyperparameters for the odd and even forms (way more modularity, hybrid combiner can be used more freely)

    typedef struct
    {
    type p_maxA_E, p_maxA_E_K, K_A_E, p_maxA_E_Z, Z_A_E;
    } p_StoredParams_Amax_E;

    typedef struct
    {
    type p_maxA_O, p_maxA_O_K, K_A_O, p_maxA_O_Z, Z_A_O;

    } p_StoredParams_Amax_O;

    typedef struct
    {
    type p_maxD_E, p_maxD_E_K, K_D_E, p_maxD_E_Z, Z_D_E;
    } p_StoredParams_Dmax_E;

    typedef struct
    {
    type p_maxD_O, p_maxD_O_K, K_D_O, p_maxD_O_Z, Z_D_O;
    } p_StoredParams_Dmax_O;

    extern p_StoredParams_Amax_E STORED_PARAMS_AE;

    extern p_StoredParams_Amax_O STORED_PARAMS_AO;

    extern p_StoredParams_Dmax_E STORED_PARAMS_DE;

    extern p_StoredParams_Dmax_O STORED_PARAMS_DO;


    extern type p_minA_E;

    extern type p_minD_E;

    extern type p_minA_O;

    extern type p_minD_O;

    #define p_max_A_E STORED_PARAMS_AE.p_maxA_E
    #define p_min_A_E p_minA_E
    #define p_max_D_E STORED_PARAMS_DE.p_maxD_E
    #define p_min_D_E p_minD_E

    #define p_max_A_O STORED_PARAMS_AO.p_maxA_O
    #define p_min_A_O p_minA_O
    #define p_max_D_O STORED_PARAMS_DO.p_maxD_O
    #define p_min_D_O p_minD_O
    /*
    type G_A_E = cast(type, 1);

    type G_A_O = cast(type, 1);

    type G_D_E = cast(type, 1);

    type G_D_O = cast(type, 1);

    #define p_G_ascEven G_A_E
    #define p_G_ascOdd G_A_O

    #define p_G_descEven G_D_E
    #define p_G_descOdd G_D_O

    */
    #define p_Z_ascEven STORED_PARAMS_AE.Z_A_E
    #define p_Z_ascOdd STORED_PARAMS_AO.Z_A_O

    #define p_K_ascEven STORED_PARAMS_AE.K_A_E
    #define p_K_ascOdd STORED_PARAMS_AO.K_A_O
    //----------------
    #define p_Z_descEven STORED_PARAMS_DE.Z_D_E
    #define p_Z_descOdd STORED_PARAMS_DO.Z_D_O

    #define p_K_descEven STORED_PARAMS_DE.K_D_E
    #define p_K_descOdd STORED_PARAMS_DO.K_D_O

    #define p_max_D_E_combine_K STORED_PARAMS_DE.p_maxD_E_K
    #define p_max_D_O_combine_K STORED_PARAMS_DO.p_maxD_O_K

    #define p_max_A_E_combine_K STORED_PARAMS_AE.p_maxA_E_K
    #define p_max_A_O_combine_K STORED_PARAMS_AO.p_maxA_O_K

    #define p_max_D_E_combine_Z STORED_PARAMS_DE.p_maxD_E_Z
    #define p_max_D_O_combine_Z STORED_PARAMS_DO.p_maxD_O_Z

    #define p_max_A_E_combine_Z STORED_PARAMS_AE.p_maxA_E_Z
    #define p_max_A_O_combine_Z STORED_PARAMS_AO.p_maxA_O_Z

    #define MAX_PARAMS_AE STORED_PARAMS_AE
    #define MAX_PARAMS_AO STORED_PARAMS_AO
    #define MAX_PARAMS_DE STORED_PARAMS_DE
    #define MAX_PARAMS_DO STORED_PARAMS_DO

    #endif //hyperparam resolution
    
#ifdef BRANCHING_SETTERS
    #ifdef ODD_EVEN_SAME_HYPERPARAMS

            void p_setMin(type min, BOOL32 Flag);

            void p_setMax(type max, BOOL32 Flag);
    /* 
            STATIC_FORCE_INLINE
            void p_setG(type g, BOOL32 Flag);
    */

            void p_setK(type k, BOOL32 Flag);

            void p_setZ(type z, BOOL32 Flag);
    #else

            void p_setMin_E(type min, BOOL32 Flag);

            void p_setMax_E(type max, BOOL32 Flag);

            void p_setMin_O(type min, BOOL32 Flag);

            void p_setMax_O(type max, BOOL32 Flag);
            /*
            STATIC_FORCE_INLINE
            void p_setG_E(type , BOOL32 Flag);
            STATIC_FORCE_INLINE
            void p_setG_O(type g, BOOL32 Flag);
            */

            void p_setK_E(type k, BOOL32 Flag);

            void p_setK_O(type k, BOOL32 Flag);


            void p_setZ_E(type z, BOOL32 Flag);

            void p_setZ_O(type z, BOOL32 Flag);
    #endif

#else //FAST_SETTERS (non-branching), default
    /*minimizes the overhead of having to pass the extra parameter to specify which base variant to change the global parameters of AND to skip the branching
    essentially making the idea of using them as learnabe parameters (that're adjusted frequently) more palpable*/

    #ifdef ODD_EVEN_SAME_HYPERPARAMS


            void p_setMinA(type min);

            void p_setMaxA(type max);

            void p_setMinD(type min);

            void p_setMaxD(type max);


            void p_setK_A(type k);

            void p_setK_D(type k);
    /*
            STATIC_FORCE_INLINE
            void p_setG_A(type g);
            STATIC_FORCE_INLINE
            void p_setG_D(type g);
    */

            void p_setZ_A(type z);

            void p_setZ_D(type z);
    #else


            void p_setMinA_E(type min);

            void p_setMinA_O(type min);


            void p_setMaxA_E(type max);

            void p_setMaxA_O(type max);


            void p_setMinD_E(type min);

            void p_setMinD_O(type min);


            void p_setMaxD_E(type max);

            void p_setMaxD_O(type max);

            void p_setK_A_E(type k);

            void p_setK_D_E(type k);

            void p_setK_A_O(type k);

            void p_setK_D_O(type k);


            void p_setZ_A_E(type z);

            void p_setZ_D_E(type z);

            void p_setZ_A_O(type z);

            void p_setZ_D_O(type z);
    #endif
#endif //BRANCHING_SETTERS
    
#ifdef __CUDACC__

    //for the ascender
    __device__ type D_pParams[12];

    #define D_p_min_A_E D_pParams[0]
    #define D_p_max_A_E_combine_K D_pParams[1]
    #define D_p_max_A_E_combine_Z D_pParams[2]

    #define D_p_min_D_E D_pParams[3]
    #define D_p_max_D_E_combine_K D_pParams[4]
    #define D_p_max_D_E_combine_Z D_pParams[5]

    #define D_p_min_A_O D_pParams[6]
    #define D_p_max_A_O_combine_K D_pParams[7]
    #define D_p_max_A_O_combine_Z D_pParams[8]

    #define D_p_min_D_O D_pParams[9]
    #define D_p_max_D_O_combine_K D_pParams[10]
    #define D_p_max_D_O_combine_Z D_pParams[11]

    #define D_p_min_A_E_idx 0
    #define D_p_max_A_E_combine_K_idx 1
    #define D_p_max_A_E_combine_Z_idx 2
    #define D_p_min_D_E_idx 3
    #define D_p_max_D_E_combine_K_idx 4
    #define D_p_max_D_E_combine_Z_idx 5
    #define D_p_min_A_O_idx 6
    #define D_p_max_A_O_combine_K_idx 7
    #define D_p_max_A_O_combine_Z_idx 8
    #define D_p_min_D_O_idx 9
    #define D_p_max_D_O_combine_K_idx 10
    #define D_p_max_D_O_combine_Z_idx 11


    /*
    the following ranges for the parameters have only been chosen by virtue
    that when the parameters are anywhere in those ranges they don't break the bounding
    behavior of the formula but they may intensify or dampen it (higher low-bound if min was negative and such)
    */

    //optimized away the G parameter in the formula to remove all extra kernel-side computations, since G was perfectly inversely proportional to K
    //but G's handling will be left commented out in case more refinement of the formula necessitates a third tunable parameter in a different position than G's original position
    /*
    STATIC_FORCE_INLINE
    type clampG(type g);
    static const type gRange[] = {cast(type, 1), MAXIMUM}; //range: [1, inf)
    */


    static const type kRange[] = {MINIMUM, MAXIMUM}; //range: (0, inf), default k = 1, at k=1 the formula has no scaling applied, k->0 increases function's velocity/sensitivty around x = 0 (which was the original behavior of g when g->infinity but g was removed in favor of k or g possesing both properties of scaling inversely to each other, g was removed because it was the parameter that couldn't be pre-computed)
    //k->infinity decreases velocity


    /* IMPORTANT NOTE: z is allowed to move in this range freely, but when 
    it becomes negative it breaks the original boundaries of the function, typically making the range transform to: [-b, a] when original range was: [a, b]
    by flipping around the y or x axis (depending on the symmetry of variant - even, odd respectively for y and x axis mirroring)
    basically z is just a multiplying factor for the equation but it CAN be zero just fine without breaking the whole equation into a constant line at y = 0; unless your minimum is absolute zero (the lib handles that by default)
    refer to the desmos link and play with z's value to see
    if you need it to always stay true to the original positive range dictated by +a and +b, just don't make it negative
    NOTE: note one thing, as z approaches -1, range of the inverses' output decreases as the whole function gradually plunges under the x-axis
    since the inverses clamp strictly following the output ranges, as z becomes more negative those output ranges are either flipped where a and b are swapped
    and/or the boundaries a and b become negated
    */
    static const type zRange[] = {cast(type, -1), cast(type, 1)}; //range: [-1, 1]


    STATIC_FORCE_INLINE
    type clampZ(type z)
    {
        /* if(z > zRange[1])
              return zRange[1];
         else if(z < zRange[0])
              return zRange[0];
         return z;
         */
        return type_min(type_max(zRange[0], z), zRange[1]);
    }

    STATIC_FORCE_INLINE //
    type clampK(type k)
    {
         return type_min(type_max(kRange[0], k), kRange[1]);
    }

    /*
    STATIC_FORCE_INLINE
    type clampG(type g)
    {
         return type_min(type_max(gRange[0], g), gRange[1]);
    }
    */
    #ifdef BRANCHING_SETTERS
        #ifdef ODD_EVEN_SAME_HYPERPARAMS

            //Flag: 1 or greater changes ascending variant's 
            //parameters, 0 changes descending variant's parameters
METALERP_DEF_SHIM_2ARG(p_setMin, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_min_D_E = SET_MIN(min, p_max_D_E);
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
                }
                else 
                {
                    p_min_A_E = SET_MIN(min, p_max_A_E);
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setMax, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_max_D_E = SET_MAX(max); ENFORCE_MAX(p_min_D_E, p_max_D_E);
                    p_max_D_E_combine_K = p_max_D_E * p_K_descEven;
                    p_max_D_E_combine_Z = p_max_D_E * p_Z_descEven;
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE_2(p_max_D_E_combine_K, p_max_D_E_combine_Z)               
                }
                else 
                {
                    p_max_A_E = SET_MAX(max); ENFORCE_MAX(p_min_A_E, p_max_A_E);
                    p_max_A_E_combine_K = p_max_A_E * p_K_ascEven;
                    p_max_A_E_combine_Z = p_max_A_E * p_Z_ascEven;
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE_2(p_max_A_E_combine_K, p_max_A_E_combine_Z)
                }
            }
            /*
            STATIC_FORCE_INLINE
            void p_setG(type g, BOOL32 Flag)
            {
                clamp_paramG(g, c_G)
                if(Flag == 0) p_G_descEven = g_Parameter;
                else p_G_ascEven = g_Parameter;
            }
            */
METALERP_DEF_SHIM_2ARG(p_setK, type, k, BOOL32, Flag)(type k, BOOL32 Flag)
            {
                clamp_paramK(k, c_K)
                if(Flag == 0) 
                {
                    p_K_descEven = k_Parameter;
                    p_max_D_E_combine_K = p_max_D_E * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
                }
                else 
                {
                    p_K_ascEven = k_Parameter;
                    p_max_A_E_combine_K = p_max_A_E * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setZ, type, z, BOOL32, Flag)(type z, BOOL32 Flag)
            {
                clamp_paramZ(z, c_Z)
                if(Flag == 0) 
                {
                    p_Z_descEven = z_Parameter;
                    p_max_D_E_combine_Z = p_max_D_E * z_Parameter;
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
                }
                else 
                {
                    p_Z_ascEven = z_Parameter;
                    p_max_A_E_combine_Z = p_max_A_E * z_Parameter;
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
                }
            }
        #else

            //Flag: 1 or greater changes ascending variant's 
            //parameters, 0 changes descending variant's parameters
METALERP_DEF_SHIM_2ARG(p_setMin_E, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_min_D_E = SET_MIN(min, p_max_D_E);
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
                }
                else 
                {
                    p_min_A_E = SET_MIN(min, p_max_A_E);
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setMax_E, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_max_D_E = SET_MAX(max); ENFORCE_MAX(p_min_D_E, p_max_D_E);
                    p_max_D_E_combine_K = p_max_D_E * p_K_descEven;
                    p_max_D_E_combine_Z = p_max_D_E * p_Z_descEven;
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE_2(p_max_D_E_combine_K, p_max_D_E_combine_Z)
                }
                else 
                {
                    p_max_A_E = SET_MAX(max); ENFORCE_MAX(p_min_A_E, p_max_A_E);
                    p_max_A_E_combine_K = p_max_A_E * p_K_ascEven;
                    p_max_A_E_combine_Z = p_max_A_E * p_Z_ascEven;
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE_2(p_max_A_E_combine_K, p_max_A_E_combine_Z)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setMin_O, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_min_D_O = SET_MIN(min, p_max_D_O);
                    clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                    COPY_PARAM2DEVICE(D_pParams, p_min_D_O)
                }
                else 
                {
                    p_min_A_O = SET_MIN(min, p_max_A_O);
                    clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                    COPY_PARAM2DEVICE(D_pParams, p_min_A_O)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setMax_O, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) 
                {
                    p_max_D_O = SET_MAX(max); ENFORCE_MAX(p_min_D_O, p_max_D_O);
                    p_max_D_O_combine_K = p_max_D_O * p_K_descOdd;
                    p_max_D_O_combine_Z = p_max_D_O * p_Z_descOdd;
                    clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                    COPY_PARAM2DEVICE_2(p_max_D_O_combine_K, p_max_D_O_combine_Z)
                }
                else 
                {
                    p_max_A_O = SET_MAX(max); ENFORCE_MAX(p_min_A_O, p_max_A_O);
                    p_max_A_O_combine_K = p_max_A_O * p_K_ascOdd;
                    p_max_A_O_combine_Z = p_max_A_O * p_Z_ascOdd;
                    clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                    COPY_PARAM2DEVICE_2(p_max_A_O_combine_K, p_max_A_O_combine_Z)
                }
            }
            /*
            STATIC_FORCE_INLINE
            void p_setG_E(type g, BOOL32 Flag)
            {
                clamp_paramG(g, c_G)
                if(Flag == 0) p_G_descEven = g_Parameter;
                else p_G_ascEven = g_Parameter;
            }
            STATIC_FORCE_INLINE
            void p_setG_O(type g, BOOL32 Flag)
            {
                clamp_paramG(g, c_G)
                if(Flag == 0) p_G_descOdd = g_Parameter;
                else p_G_ascOdd = g_Parameter;
            }
            */
METALERP_DEF_SHIM_2ARG(p_setK_E, type, k, BOOL32, Flag)(type k, BOOL32 Flag)
            {
                clamp_paramK(k, c_K)
                if(Flag == 0) 
                {
                    p_K_descEven = k_Parameter;
                    p_max_D_E_combine_K = p_max_D_E * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
                }
                else 
                {
                    p_K_ascEven= k_Parameter;
                    p_max_A_E_combine_K = p_max_A_E * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setK_O, type, k, BOOL32, Flag)(type k, BOOL32 Flag)
            {
                clamp_paramK(k, c_K)
                if(Flag == 0)
                {
                    p_K_descOdd = k_Parameter;
                    p_max_D_O_combine_K = p_max_D_O * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_K)
                }
                else 
                {
                    p_K_ascOdd = k_Parameter;
                    p_max_A_O_combine_K = p_max_A_O * k_Parameter;
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_K)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setZ_E, type, z, BOOL32, Flag)(type z, BOOL32 Flag)
            {
                clamp_paramZ(z, c_Z)
                if(Flag == 0) 
                {
                    p_Z_descEven = z_Parameter;
                    p_max_D_E_combine_Z = p_max_D_E * z_Parameter;
                    clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
                }
                else 
                {
                    p_Z_ascEven = z_Parameter;
                    p_max_A_E_combine_Z = p_max_A_E * z_Parameter;
                    clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
                }
            }
METALERP_DEF_SHIM_2ARG(p_setZ_O, type, z, BOOL32, Flag)(type z, BOOL32 Flag)
            {
                clamp_paramZ(z, c_Z)
                if(Flag == 0) 
                {
                    p_Z_descOdd = z_Parameter;
                    p_max_D_O_combine_Z = p_max_D_O * z_Parameter;
                    clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                    COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_Z)
                }
                else 
                {
                    p_Z_ascOdd = z_Parameter;
                    p_max_A_O_combine_Z = p_max_A_O * z_Parameter;
                    clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                    COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_Z)
                }
            }

        #endif

    #else //FAST_SETTERS (non-branching), default
    /*minimizes the overhead of having to pass the extra parameter to specify which base variant to change the global parameters of AND to skip the branching
    essentially making the idea of using them as learnabe parameters (that're adjusted frequently) more palpable*/

        #ifdef ODD_EVEN_SAME_HYPERPARAMS

METALERP_DEF_SHIM(p_setMinA, type, min)(type min)
            {
                p_min_A_E = SET_MIN(min, p_max_A_E);
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
            }
METALERP_DEF_SHIM(p_setMaxA, type, max)(type max) 
            {
                p_max_A_E = SET_MAX(max); ENFORCE_MAX(p_min_A_E, p_max_A_E);
                p_max_A_E_combine_K = p_max_A_E * p_K_ascEven;
                p_max_A_E_combine_Z = p_max_A_E * p_Z_ascEven;
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE_2(p_max_A_E_combine_K, p_max_A_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setMinD, type, min)(type min)
            {
                p_min_D_E = SET_MIN(min, p_max_D_E);
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
            }
METALERP_DEF_SHIM(p_setMaxD, type, max)(type max)
            {
                p_max_D_E = SET_MAX(max); ENFORCE_MAX(p_min_D_E, p_max_D_E);
                p_max_D_E_combine_K = p_max_D_E * p_K_descEven;
                p_max_D_E_combine_Z = p_max_D_E * p_Z_descEven;
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE_2(p_max_D_E_combine_K, p_max_D_E_combine_Z)
            }

METALERP_DEF_SHIM(p_setK_A, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_ascEven = k_Parameter;
                p_max_A_E_combine_K = p_max_A_E * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
            }
METALERP_DEF_SHIM(p_setK_D, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_descEven = k_Parameter;
                p_max_D_E_combine_K = p_max_D_E * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
            }
            /*
            STATIC_FORCE_INLINE
            void p_setG_A(type g)
            {
                clamp_paramG(g, c_G)
                p_G_ascEven = g_Parameter;
            }
            STATIC_FORCE_INLINE
            void p_setG_D(type g)
            {
                clamp_paramG(g, c_G)
                p_G_descEven = g_Parameter;
            }
            */
METALERP_DEF_SHIM(p_setZ_A, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_ascEven = z_Parameter;
                p_max_A_E_combine_Z = p_max_A_E * z_Parameter;
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setZ_D, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_descEven = z_Parameter;
                p_max_D_E_combine_Z = p_max_D_E * z_Parameter;
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
            }
        #else

METALERP_DEF_SHIM(p_setMinA_E, type, min)(type min) 
            { 
                p_min_A_E = SET_MIN(min, p_max_A_E); 
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE(D_pParams, p_min_A_E)
            }
METALERP_DEF_SHIM(p_setMinA_O, type, min)(type min) 
            { 
                p_min_A_O = SET_MIN(min, p_max_A_O); 
                clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                COPY_PARAM2DEVICE(D_pParams, p_min_A_O)
            }

METALERP_DEF_SHIM(p_setMaxA_E, type, max)(type max) 
            {
                p_max_A_E = SET_MAX(max); ENFORCE_MAX(p_min_A_E, p_max_A_E);
                p_max_A_E_combine_K = p_K_ascEven * p_max_A_E;
                p_max_A_E_combine_Z = p_Z_ascEven * p_max_A_E;
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE_2(D_pParams, p_max_A_E_combine_K, p_max_A_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setMaxA_O, type, max)(type max) 
            {
                p_max_A_O = SET_MAX(max); ENFORCE_MAX(p_min_A_O, p_max_A_O);
                p_max_A_O_combine_K = p_K_ascOdd * p_max_A_O;
                p_max_A_O_combine_Z = p_Z_ascOdd * p_max_A_O;
                clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                COPY_PARAM2DEVICE_2(D_pParams, p_max_A_O_combine_K, p_max_A_O_combine_Z)
            }
METALERP_DEF_SHIM(p_setMinD_E, type, min)(type min) 
            { 
                p_min_D_E = SET_MIN(min, p_max_D_E); 
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE(D_pParams, p_min_D_E)
            }
METALERP_DEF_SHIM(p_setMinD_O, type, min)(type min) 
            { 
                p_min_D_O = SET_MIN(min, p_max_D_O); 
                clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                COPY_PARAM2DEVICE(D_pParams, p_min_D_O)
            }

METALERP_DEF_SHIM(p_setMaxD_E, type, max)(type max) 
            {
                p_max_D_E = SET_MAX(max); ENFORCE_MAX(p_min_D_E, p_max_D_E);
                p_max_D_E_combine_K = p_K_descEven * p_max_D_E;
                p_max_D_E_combine_Z = p_Z_descEven * p_max_D_E;
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE_2(D_pParams, p_max_D_E_combine_K, p_max_D_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setMaxD_O, type, max)(type max) 
            {
                p_max_D_O = SET_MAX(max); ENFORCE_MAX(p_min_D_O, p_max_D_O);
                p_max_D_O_combine_K = p_K_descOdd * p_max_D_O;
                p_max_D_O_combine_Z = p_Z_descOdd * p_max_D_O;
                clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                COPY_PARAM2DEVICE_2(D_pParams, p_max_D_O_combine_K, p_max_D_O_combine_Z)
            }

            /*STATIC_FORCE_INLINE
            void p_setG_A_E(type g);
            STATIC_FORCE_INLINE
            void p_setG_D_E(type g);
            STATIC_FORCE_INLINE
            void p_setG_A_O(type g);
            STATIC_FORCE_INLINE
            void p_setG_D_O(type g);*/

            /*
            STATIC_FORCE_INLINE
            void p_setG_A_E(type g)
            {
                clamp_paramG(g, c_G)
                p_G_ascEven = g_Parameter;
            }
            STATIC_FORCE_INLINE
            void p_setG_D_E(type g)
            {
                clamp_paramG(g, c_G)
                p_G_descEven = g_Parameter;
            }
            STATIC_FORCE_INLINE
            void p_setG_A_O(type g)
            {
                clamp_paramG(g, c_G)
                p_G_ascOdd = g_Parameter;
            }
            STATIC_FORCE_INLINE
            void p_setG_D_O(type g)
            {
                clamp_paramG(g, c_G)
                p_G_descOdd = g_Parameter;
            }
            */

METALERP_DEF_SHIM(p_setK_A_E, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_ascEven = k_Parameter;
                p_max_A_E_combine_K = p_max_A_E * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_K)
            }
METALERP_DEF_SHIM(p_setK_D_E, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_descEven = k_Parameter;
                p_max_D_E_combine_K = p_max_D_E * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_K)
            }
METALERP_DEF_SHIM(p_setK_A_O, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_ascOdd = k_Parameter;
                p_max_A_O_combine_K = p_max_A_O * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_K)
            }
METALERP_DEF_SHIM(p_setK_D_O, type, k)(type k)
            {
                clamp_paramK(k, c_K)
                p_K_descOdd = k_Parameter;
                p_max_D_O_combine_K = p_max_D_O * k_Parameter;
                COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_K)
            }

METALERP_DEF_SHIM(p_setZ_A_E, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_ascEven = z_Parameter;
                p_max_A_E_combine_Z = p_max_A_E * z_Parameter;
                clampZMax(p_max_A_E_combine_Z, p_min_A_E)
                COPY_PARAM2DEVICE(D_pParams, p_max_A_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setZ_D_E, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_descEven = z_Parameter;
                p_max_D_E_combine_Z = p_max_D_E * z_Parameter;
                clampZMax(p_max_D_E_combine_Z, p_min_D_E)
                COPY_PARAM2DEVICE(D_pParams, p_max_D_E_combine_Z)
            }
METALERP_DEF_SHIM(p_setZ_A_O, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_ascOdd = z_Parameter;
                p_max_A_O_combine_Z = p_max_A_O * z_Parameter;
                clampZMax(p_max_A_O_combine_Z, p_min_A_O)
                COPY_PARAM2DEVICE(D_pParams, p_max_A_O_combine_Z)
            }
METALERP_DEF_SHIM(p_setZ_D_O, type, z)(type z)
            {
                clamp_paramZ(z, c_Z)
                p_Z_descOdd = z_Parameter;
                p_max_D_O_combine_Z = p_max_D_O * z_Parameter;
                clampZMax(p_max_D_O_combine_Z, p_min_D_O)
                COPY_PARAM2DEVICE(D_pParams, p_max_D_O_combine_Z)
            }

        #endif
    #endif //BRANCHING_SETTERS

#endif //cuda compiled interface funcs

#if defined(METALERP_REDEF_LATER) && defined(INSIDE_COMBOS_H)

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


#define p_min_A_E NM(p_min_A_E)
#define p_max_A_E_combine_K NM(p_max_A_E_combine_K)
#define p_max_A_E_combine_Z NM(p_max_A_E_combine_Z)

#define p_min_D_E NM(p_min_D_E)
#define p_max_D_E_combine_K NM(p_max_D_E_combine_K)
#define p_max_D_E_combine_Z NM(p_max_D_E_combine_Z)

#define p_min_A_O NM(p_min_A_O)
#define p_max_A_O_combine_K NM(p_max_A_O_combine_K)
#define p_max_A_O_combine_Z NM(p_max_A_O_combine_Z)

#define p_min_D_O NM(p_min_D_O)
#define p_max_D_O_combine_K NM(p_max_D_O_combine_K)
#define p_max_D_O_combine_Z NM(p_max_D_O_combine_Z)

#endif
//this is the "simplified" or more compacted form of the functions, the lib implements the equivalent LERP form of the equation because they're more optimizable
/*    extra multiplications, but optimized away with pre-computations to match the base forms' speed
    ascender:
     b * (z * |x| + k a)
y = ________________________ -> z*b precomputed, k*b precomputed, equivalent lerp form used: |x|/(|x|+kb) * (zb-a) + a
      k * b   +  |x|

      descender:
     z * k * b^2 + a * |x|
y = ___________________________  equivalent lerp form used:  kb/(|x|+kb) * (zb-a) + a
        k * b   +  |x|
*/

METALERP_INTERNAL_DEVFUNC type NM(p_ascendingVariant_E)(type x);

METALERP_INTERNAL_DEVFUNC type NM(p_ascendingVariant_O)(type x);

METALERP_INTERNAL_DEVFUNC type NM(p_descendingVariant_E)(type x); 

METALERP_INTERNAL_DEVFUNC type NM(p_descendingVariant_O)(type x);


METALERP_INTERNAL_DEVFUNC type NM(p_ascendingV_O_RightArm)(type x);
METALERP_INTERNAL_DEVFUNC type NM(p_ascendingV_O_LeftArm)(type x);

METALERP_INTERNAL_DEVFUNC type NM(p_descendingV_O_RightArm)(type x);
METALERP_INTERNAL_DEVFUNC type NM(p_descendingV_O_LeftArm)(type x);

//clamp functions
METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_Even)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_Even)(type y);

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_R)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_L)(type y);

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_R)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_L)(type y);

//inverses
METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingVariant_E)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingVariant_E)(type y); 


METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingVariant_O)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingV_O_RightArm)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingV_O_LeftArm)(type y);

METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingVariant_O)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingV_O_RightArm)(type y);
METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingV_O_LeftArm)(type y);

/*implementations:*/
METALERP_INTERNAL_KERNEL(p_ascendingVariant_E)(type x)
{
    #ifdef NO_ABS
        #define p_A_absVar x
    #else
        type p_A_absVar = type_abs(x); 
    #endif   
    
    type t = p_A_absVar/(p_A_absVar + p_max_A_E_combine_K);
    return LERP(p_min_A_E, p_max_A_E_combine_Z, t);
}
METALERP_INTERNAL_KERNEL(p_ascendingVariant_O)(type x)
{
    #ifndef NO_ABS
    type p_A_absVar = type_abs(x);
    #endif

    type t = p_A_absVar/(p_A_absVar + p_max_A_O_combine_K);
    return LERP(p_min_A_O, p_max_A_O_combine_Z, t) * SGN(x);
}
METALERP_INTERNAL_KERNEL(p_descendingVariant_E)(type x) 
{
   #ifdef NO_ABS
        #define p_D_absVar x
    #else
        #define p_D_absVar type_abs(x) 
    #endif
    type t = p_max_D_E_combine_K/(p_D_absVar + p_max_D_E_combine_K);
    return LERP(p_min_D_E, p_max_D_E_combine_Z, t);
}
METALERP_INTERNAL_KERNEL(p_descendingVariant_O)(type x)
{
    type t = p_max_D_O_combine_K/(p_D_absVar + p_max_D_O_combine_K);
    return LERP(p_min_D_E, p_max_D_O_combine_Z, t) * SGN(x);
}

/*
METALERP_INTERNAL_KERNEL(p_ascendingVariant_O)(type x)
{
    return( EXPECT_BRANCH(x ASC_POS_COMPARISON cast(type, 0)) ?
     p_ascendingV_O_RightArm(x) : p_ascendingV_O_LeftArm(x) );
}
METALERP_INTERNAL_KERNEL(p_descendingVariant_O)(type x)
{
    return( EXPECT_BRANCH(x DESC_POS_COMPARISON cast(type, 0)) ?
     p_descendingV_O_RightArm(x) : p_descendingV_O_LeftArm(x) );
}
*/

METALERP_INTERNAL_DEVFUNC type NM(p_ascendingV_O_RightArm)(type x)
{
    type t = x / (x + p_max_A_O_combine_K );
    return LERP(p_min_A_O, p_max_A_O_combine_Z, t);
}
METALERP_INTERNAL_DEVFUNC type NM(p_ascendingV_O_LeftArm)(type x)
{
    type t = x / (p_max_A_O_combine_K - x);
    return minusMinLERP(p_min_A_O, p_max_A_O_combine_Z, t);
}
METALERP_INTERNAL_DEVFUNC type NM(p_descendingV_O_RightArm)(type x)
{
    type t = p_max_D_O_combine_K / (x + p_max_D_O_combine_K);
    return LERP(p_min_D_O, p_max_D_O_combine_Z, t);
}
METALERP_INTERNAL_DEVFUNC type NM(p_descendingV_O_LeftArm)(type x)
{   
    type t = p_max_D_O_combine_K / (x - p_max_D_O_combine_K);
    return minusMinLERP(p_min_D_O, p_max_D_O_combine_Z, t);
}

/*****************/

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_Even)(type y)
{   //stays inside (a, b) open interval, because equalling either one of those parameters can result in division by zero in the inverse formulas (both ascender and descender can suffer from this)
    //the value will certainly be comically large with such a small denominator that results from subtractions like: (5 - 4.9...) in the denom but atleast it won't be a zero in the denom
    
    return p_max_A_E_combine_Z > p_min_A_E ? //can never be equal to it since the lib internally handles this case in the setters
    type_min(type_max(y, OPEN_INTERVAL_MIN(p_min_A_E)), OPEN_INTERVAL_MAX(p_max_A_E_combine_Z)) :
    type_min(type_max(y, OPEN_INTERVAL_MIN(p_max_A_E_combine_Z)), OPEN_INTERVAL_MAX(p_min_A_E));
}

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_Even)(type y)
{
    return p_max_D_E_combine_Z > p_min_D_E ? //can never be equal to it since the lib internally handles this case in the setters
    type_min(type_max(y, OPEN_INTERVAL_MIN(p_min_D_E)), OPEN_INTERVAL_MAX(p_max_D_E_combine_Z)) :
    type_min(type_max(y, OPEN_INTERVAL_MIN(p_max_D_E_combine_Z)), OPEN_INTERVAL_MAX(p_min_D_E));
}

//odd, these have potential for more optimization, especially since I've gone with the more determinable route of heavier conditions instead of heavy redundant arithmetic
METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_R)(type y)
{
    if( p_min_A_O ASC_POS_COMPARISON cast(type, 0) )
    {
        if(p_max_A_O_combine_Z > p_min_A_O)
            return
            type_min(type_max(y, OPEN_INTERVAL_MIN(p_min_A_O)), OPEN_INTERVAL_MAX(p_max_A_O_combine_Z));
        else
            return
            type_min(type_max(y, type_max(OPEN_INTERVAL_MIN(cast(type, 0)), OPEN_INTERVAL_MIN(p_max_A_O_combine_Z)) ),
             OPEN_INTERVAL_MAX(p_min_A_O));
    }
    else
    {
        if(p_max_A_O_combine_Z > p_min_A_O)
            return
            type_min(type_max(-y, OPEN_INTERVAL_MIN(p_min_A_O)), 
            type_min(OPEN_INTERVAL_MAX(cast(type, 0)), OPEN_INTERVAL_MAX(p_max_A_O_combine_Z)));
        else
            return
            type_min(type_max(-y, OPEN_INTERVAL_MIN(p_max_A_O_combine_Z)), OPEN_INTERVAL_MAX(p_min_A_O));
    }
}

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_A_L)(type y)
{
    if( p_min_A_O ASC_POS_COMPARISON cast(type, 0) )
    {
        if(p_max_A_O_combine_Z > p_min_A_O)
            return
            type_min(type_max(y, -OPEN_INTERVAL_MIN(p_max_A_O_combine_Z)), -OPEN_INTERVAL_MAX(p_min_A_O));
        else
            return
            type_min( type_max(y, -OPEN_INTERVAL_MIN(p_min_A_O)),
            type_min(OPEN_INTERVAL_MAX(cast(type, 0)), -OPEN_INTERVAL_MAX(p_max_A_O_combine_Z)) );
    }
    else
    {
        if(p_max_A_O_combine_Z > p_min_A_O)
            return
            type_min(type_max(-y, type_max(OPEN_INTERVAL_MIN(cast(type, 0)), -OPEN_INTERVAL_MIN(p_max_A_O_combine_Z))),
            -OPEN_INTERVAL_MAX(p_min_A_O) );
        else
            return
            type_min(type_max(-y, -OPEN_INTERVAL_MIN(p_min_A_O) ),
            -OPEN_INTERVAL_MAX(p_max_A_O_combine_Z));
    }
}

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_R)(type y)
{
    // p_max_D_O_combine_Z's sign is same as p_Z_descOdd, on the assumption that p_max_D_O is always positive, which the lib enforces by default.
    //this allows to use the fused hyperparameter (e.g. zb or kb) in checks that're mainly sensitive to the z parameter's sign (K cannot become negative by default)
    //which means we have to store and worry about less variables on the GPU
    
    if( p_max_D_O_combine_Z DESC_POS_COMPARISON cast(type, 0) )
    {
        if(p_max_D_O_combine_Z > p_min_D_O)
            return
            type_min(type_max(y, type_max(OPEN_INTERVAL_MIN(cast(type, 0)), OPEN_INTERVAL_MIN(p_min_D_O)) ),
            OPEN_INTERVAL_MAX(p_max_D_O_combine_Z));
        else
            return
            type_min(type_max(y, OPEN_INTERVAL_MIN(p_max_D_O_combine_Z)),
            OPEN_INTERVAL_MAX(p_min_D_O));
    }
    else
    {
        if(p_max_D_O_combine_Z > p_min_D_O)
            return
            type_min(type_max(-y, OPEN_INTERVAL_MIN(p_min_D_O)), 
            OPEN_INTERVAL_MAX(p_max_D_O_combine_Z));
        else
            return
            type_min(type_max(-y, OPEN_INTERVAL_MIN(p_max_D_O_combine_Z)),
            type_min(OPEN_INTERVAL_MAX(cast(type, 0)), OPEN_INTERVAL_MAX(p_min_D_O)) );
    }
}

METALERP_INTERNAL_DEVFUNC type NM(p_clampY_D_L)(type y)
{
    if( p_max_D_O_combine_Z DESC_POS_COMPARISON cast(type, 0) )
    {
        if(p_max_D_O_combine_Z > p_min_D_O)
            return
            type_min(type_max(y, -OPEN_INTERVAL_MIN(p_max_D_O_combine_Z) ),
            type_min(OPEN_INTERVAL_MAX(cast(type, 0)), -OPEN_INTERVAL_MAX(p_min_D_O)) );
        else
            return
            type_min(type_max(y, -OPEN_INTERVAL_MIN(p_min_D_O)),
            -OPEN_INTERVAL_MAX(p_max_D_O_combine_Z) );
    }
    else
    {
        if(p_max_D_O_combine_Z > p_min_D_O)
            return
            type_min(type_max(y, -OPEN_INTERVAL_MIN(p_max_D_O_combine_Z) ), 
            -OPEN_INTERVAL_MAX(p_min_D_O));
        else
            return
            type_min(type_max(-y, type_max(OPEN_INTERVAL_MIN(cast(type, 0)), -OPEN_INTERVAL_MIN(p_min_D_O)) ),
            -OPEN_INTERVAL_MAX(p_max_D_O_combine_Z));
    }
}

/*****************/

METALERP_INTERNAL_KERNEL(p_inv_ascendingVariant_E)(type y)
{
    type yClamped = NM(p_clampY_A_Even)(y);
    return type_fma(p_max_A_E_combine_K, yClamped, 
        -(p_max_A_E_combine_K * p_min_A_E)) 
        / (p_max_A_E_combine_Z - yClamped);
}

METALERP_INTERNAL_KERNEL(p_inv_descendingVariant_E)(type y)
{
    type yClamped = NM(p_clampY_D_Even)(y);
    return type_fma(p_max_D_E_combine_K, p_max_D_E_combine_Z,
        -(p_max_D_E_combine_K * yClamped)) 
        / (yClamped - p_min_D_E);
}


METALERP_INTERNAL_KERNEL(p_inv_ascendingVariant_O)(type y)
{
    return NBSGN(p_min_A_O) * ((y ASC_POS_COMPARISON 0) ?
     NM(p_inv_ascendingV_O_RightArm)(y) :
     NM(p_inv_ascendingV_O_LeftArm)(y)) ;
}

METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingV_O_RightArm)(type y)
{
    type yClamped = NM(p_clampY_A_R)(y);
  //  printf("\n-------------------\nPAO-yclamped(R): %.20f\n-------------------\n", yClamped);
    return type_fma(p_max_A_O_combine_K, yClamped, 
        -(p_max_A_O_combine_K * p_min_A_O)) 
        / (p_max_A_O_combine_Z - yClamped);
}
METALERP_INTERNAL_DEVFUNC type NM(p_inv_ascendingV_O_LeftArm)(type y)
{
    type yClamped = NM(p_clampY_A_L)(y);
   /// printf("\n-------------------\nPAO-yclamped(L): %.20f\n-------------------\n", yClamped);
    return type_fma(p_max_A_O_combine_K, yClamped, 
        (p_max_A_O_combine_K * p_min_A_O)) 
        / (p_max_A_O_combine_Z + yClamped);
}

METALERP_INTERNAL_KERNEL(p_inv_descendingVariant_O)(type y)
{   
    return NBSGN(p_max_D_O_combine_Z) * ((y DESC_POS_COMPARISON 0) ?
     NM(p_inv_descendingV_O_RightArm)(y) : 
     NM(p_inv_descendingV_O_LeftArm)(y));
}

//(kbzb - kby) / (y - a)
METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingV_O_RightArm)(type y)
{
    type yClamped = NM(p_clampY_D_R)(y);
  //printf("\n-------------------\nPDO-yclamped(R): %.20f\n-------------------\n", yClamped);
    return type_fma(p_max_D_O_combine_K, p_max_D_O_combine_Z,
        -(p_max_D_O_combine_K * yClamped)) 
        / (yClamped - p_min_D_O);
}
//(kbzb + kby) / (y + a)
METALERP_INTERNAL_DEVFUNC type NM(p_inv_descendingV_O_LeftArm)(type y)
{
    type yClamped = NM(p_clampY_D_L)(y);
 //  printf("\n-------------------\nPDO-yclamped(L): %.20f\n-------------------\n", yClamped);
    return type_fma(p_max_D_O_combine_K, p_max_D_O_combine_Z,
        (p_max_D_O_combine_K * yClamped)) 
        / (yClamped + p_min_D_O);
}

/*
METALERP_INTERNAL_KERNEL
type p_inv_descendingV_O_RightArm2(type y)
{
    type yClamped = p_clampY_D_R(y);
  //  printf("\n-------------------\nPDO-yclamped(R): %.20f\n-------------------\n", yClamped);
    return p_max_D_O_combine_K * (p_max_D_O_combine_Z - yClamped) / (yClamped - p_min_D_O);
}
//(kbzb + kby) / (y + a)
METALERP_INTERNAL_KERNEL
type p_inv_descendingV_O_LeftArm2(type y)
{
    type yClamped = p_clampY_D_L(y);
   // printf("\n-------------------\nPDO-yclamped(L): %.20f\n-------------------\n", yClamped);
    return p_max_D_O_combine_K * (p_max_D_O_combine_Z + yClamped) / (yClamped + p_min_D_O);
}
METALERP_INTERNAL_KERNEL
type p_inv_descendingVariant_O2(type y)
{
    return( EXPECT_BRANCH(y DESC_POS_COMPARISON 0) ?
     p_inv_descendingV_O_RightArm2(y) : p_inv_descendingV_O_LeftArm2(y) );
}
     */


#endif