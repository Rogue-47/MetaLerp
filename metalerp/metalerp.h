
/*
Metalerp C/C++ Interface

proper download of this library from the github repository should give you a copy of the LGPL license text file
that comes with it, if the license was not attained along with this source code; see: (PLACEHOLDER)


This library twists linear interpolation into a fast and powerful
family of transformative mathematical behavior approximators/emulators and constructors
with a very simple interface of at most 2 hyperparameters (for the base forms)
or 4 total hyperparameters for the parametric (in advancedForms.h) forms 
think of it as an implementation of a kind of fast (purely-arithmetic) **non-linear interpolation**.

**************LIGHTWEIGHT LIB HEADER FOR META-LERP**************
you need only to include this header and link with the libmetalerp static lib using -lmetalerp to get the full release
functionality of meta-lerp in C/CPP

you can construct your own new behaviors with visual aid from the desmos 
sheet at: https://www.desmos.com/calculator/cf0389db8e (for messing with parameters and seeing how the functions change) --editor note: link will change most likely


also check out all current approximations I made up until now
at: https://www.desmos.com/calculator/neji45kf1n (you can add more approximators using the lib's functions and parameters yourself in your own local copy,
 please share any cool findings you come across) --editor note: link will change most likely

All ideas/formulations/approximations/implementation/benchmarking/demonstrations
behind MetaLerp up until v1.0 are the work of Omar M. Mahmoud

credited external dependencies:
python C API (for the python bindings interface)
numpy (for ndarray object definition and compatibility with numpy arrays and operations)
xoshiro256+ and xoshiro256++ (for PRNGs, at: https://prng.di.unimi.it)


- Rogue-47/Omar, August, 2025 - v1.0
*/

//NOTE: define METALERP_RELEASE macro before including this header to make the lib switch from assertions to more tolerant error handling
//define METALERP_FAST to make the lib run as fast as it can with only the basic mathematical and boundary violation checks still in place

//NOTE: call this macro (or the function call it expands to) at the beginning of main to initialize the lib's global state correctly:
#ifndef METALERP_INIT
#define METALERP_INIT metalerp_init();
#endif

/*critical:
it is best you compile the program using metalerp's functionality with the same performance macros that you
compiled the lib with, so if the lib was compiled with -DMETALERP_FAST (for the nvcc compilation phase and gcc 
(can do -D definition too or just define it in program code before including metalerp.h))
you HAVE to compile the program with that too to not get any signature mismatch-resultant errors with the binary definitions
same with METALERP_RELEASE, if you did not specify any macros explicitly in the compilation command,
you do not have to define them for the program.

same goes for choosing the native computation type of metalerp, if you explicitly specified it as double you
have to specify it as double in the main program's translation unit and other files which include metalerp lib*/

#if !defined(LIB_METALERP) && !defined(META_LERP)
#define LIB_METALERP

    //full lib interface functionality
    #define METALERP_INTERFACE_LIBMODE

    #ifdef __cplusplus
        extern "C" {
    #endif
        #include "core/include/headers/commons.h" 
        
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
        static const uint32_t HYBRID_TABLE_SIZE = P_INV_DESC_ODD + 1;


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
        static const uint32_t HYBRID_LR_TABLE_SIZE = LR_P_INV_DESC_ODD_R + 1;

        #if !defined(METALERP_RELEASE) && !defined(METALERP_FAST)
            #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName)  void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t in_len, const size_t out_len)
        #elif !defined(METALERP_FAST) //higher tolerance, for production environments
            #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName)  void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t in_len, const size_t out_len)
        #else //fastest, used for maximum speed, but also vulnerable if memory/bounds are not checked correctly before the function is called
       
            #define METALERP_DEF_OUTER_DISPATCHER(dispatcherName)  void  \
                dispatcherName(const type* __restrict__ in, type* __restrict__ out, const size_t len)
        #endif

        /*common header defines [void setSignBias(type num)] for setting the lib's sgn function's 
        sign bias on absolute zero, it is callable like the rest below*/
        /*lib state interface*/
        void metalerp_init();
        void setCUDA_Mode(BOOL32 num);
        BOOL32 getCUDA_Mode();
        void setSignBias(type num);

        /*approximation layer*/
        STATIC_FORCE_INLINE type
        Sigmack(type x);
        void setSigmackParams(type min, type max, type k, type v);
        void resetSigmackParams();

        STATIC_FORCE_INLINE type
        NormDistApproximator(type x);
        void setNormDistParams(type standardDeviation, type mean);
        void setNormDistTunableParams(type vParam, type minFactor, type maxFactor, type kFactor);
        void resetNormDistParams();

        /*hybrid arm combiner*/
        STATIC_FORCE_INLINE type
        hybridVariant(type x);
        
        STATIC_FORCE_INLINE type
        hybridVariant_LR(type x);

        void setHybridComboArms(enum Functions L_Arm, enum Functions R_Arm);
        void setHybridComboArms_LR(enum Functions_LR L_Arm, enum Functions_LR R_Arm);

        /*base (non-parametric) variants*/
        void setMinA_E(type min);
        void setMinA_O(type min);
        void setMaxA_E(type max);
        void setMaxA_O(type max);
        void setMinD_E(type min);
        void setMinD_O(type min);
        void setMaxD_E(type max);
        void setMaxD_O(type max);

        STATIC_FORCE_INLINE type 
        ascendingVariant_E(type x);
        
        STATIC_FORCE_INLINE type 
        descendingVariant_O(type x);
        
        STATIC_FORCE_INLINE type 
        descendingVariant_E(type x);
        
        STATIC_FORCE_INLINE type 
        ascendingVariant_O(type x);

        STATIC_FORCE_INLINE type 
        inv_ascendingVariant_E(type y);
        
        STATIC_FORCE_INLINE type 
        inv_ascendingVariant_O(type y);

        STATIC_FORCE_INLINE type 
        inv_descendingVariant_E(type y);
        
        STATIC_FORCE_INLINE type 
        inv_descendingVariant_O(type y);

        /*parametric variants*/
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

        STATIC_FORCE_INLINE type
        p_ascendingVariant_E(type x);

        STATIC_FORCE_INLINE type
        p_ascendingVariant_O(type x);
            
        STATIC_FORCE_INLINE type
        p_descendingVariant_E(type x);
            
        STATIC_FORCE_INLINE type
        p_descendingVariant_O(type x);
            
        STATIC_FORCE_INLINE type
        p_inv_ascendingVariant_E(type y);
            
        STATIC_FORCE_INLINE type
        p_inv_ascendingVariant_O(type y);

        STATIC_FORCE_INLINE type
        p_inv_descendingVariant_E(type y);
            
        STATIC_FORCE_INLINE type
        p_inv_descendingVariant_O(type y);

        /*batched processing interface*/
        //approximators
        METALERP_DEF_OUTER_DISPATCHER(batched_Sigmack);
        METALERP_DEF_OUTER_DISPATCHER(batched_NormDistApproximator);
        
        //hybrid arm-combiners (can also pick inverses from them)
        METALERP_DEF_OUTER_DISPATCHER(batched_Hybrid);
        METALERP_DEF_OUTER_DISPATCHER(batched_Hybrid_LR);
        
        //base variants
        METALERP_DEF_OUTER_DISPATCHER(batched_B_A_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_B_A_O);
        METALERP_DEF_OUTER_DISPATCHER(batched_B_D_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_B_D_O);
        //base variants inverses
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_A_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_A_O);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_D_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_B_D_O);
        
        //parametric variants
        METALERP_DEF_OUTER_DISPATCHER(batched_P_A_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_P_A_O);
        METALERP_DEF_OUTER_DISPATCHER(batched_P_D_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_P_D_O);
        //parametric variants inverses
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_A_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_A_O);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_D_E);
        METALERP_DEF_OUTER_DISPATCHER(batched_inv_P_D_O);

    #ifdef __cplusplus
        }
    #endif

    #undef METALERP_INTERFACE_LIBMODE

#endif



#ifndef INCLUDE_METALERP_INTERNAL_MACRO_UTILS //define before including metalerp.h if there's need for the utility macros of metalerp
    
    #ifdef cast
        #undef cast
    #endif
    
    #ifdef type
        #undef type
    #endif
    #ifdef type_abs
        #undef type_abs
    #endif
    #ifdef type_fma
        #undef type_fma
    #endif
    #ifdef type_min
        #undef type_min
    #endif
    #ifdef type_max
        #undef type_max
    #endif
    #ifdef MINIMUM
        #undef MINIMUM
    #endif
    #ifdef MAXIMUM
        #undef MAXIMUM
    #endif
    #ifdef NM
    #undef NM
    #endif

    /*not re-defined by the commons header*/
    #ifdef clamp_paramK
    #undef clamp_paramK
    #endif

    #ifdef k_Parameter
    #undef k_Parameter
    #endif

    #ifdef clamp_paramZ
    #undef clamp_paramZ
    #endif

    #ifdef z_Parameter
    #undef z_Parameter
    #endif

    #ifdef SGN
    #undef SGN
    #endif

    #ifdef BSGN
    #undef BSGN
    #endif

    #ifdef NBSGN
    #undef NBSGN
    #endif

#endif

