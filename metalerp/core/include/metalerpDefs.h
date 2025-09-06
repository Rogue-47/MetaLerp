/*
proper download of this library from the github repository should give you a copy of the LGPL license text file
that comes with it, if the license was not attained along with this source code; see: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt


This library twists linear interpolation into a fast and powerful
family of transformative mathematical behavior approximators/emulators and building blocks
with a very simple interface of at most 2 hyperparameters (min and max) (for the base forms)
or 4 total hyperparameters for the parametric (in advancedForms.h) forms (min, max, k, z (also called v in the approximator interface)) 
think of it as an implementation of a kind of fast (purely-arithmetic) **non-linear interpolation**.

**************ENTRY HEADER FOR META-LERP**************
you need only to include this header to get the full release
functionality of meta-lerp in C/CPP

you can construct your own new behaviors with visual aid from the desmos 
sheet at: https://www.desmos.com/calculator/cf0389db8e (for messing with parameters and seeing how the functions change) --editor note: link will change most likely


also check out all current approximations I made up until now
at: https://www.desmos.com/calculator/neji45kf1n (you can add more approximators using the lib's functions and parameters yourself in your own local copy,
 please share any cool findings you come across)

credits:
author: Omar M. Mahmoud

external libs used:
for the core lib:
xoshiro256+ and xoshiro256++ (for PRNGs, at: https://prng.di.unimi.it)

for the python interface:
python C API
numpy (for operability with numpy arrays)


- Rogue-47/Omar, September, 2025 - v1.0
for proper initialization of the lib in C/C++, type METALERP_INIT at the beginning of main
*/


/**************************************************/
//TODO (most important): (base and advanced, already applied in approximator) (kernel/setter) compute b-a as a fused var in the setters
//TODO: (lib-wide) () add CUDA-capability independence later with a macro that enables compiling the source code to only account for microprocessor code
//TODO: (in kDispatcher) (dispatcher) make the function that checks the hybrid arrays to determine exactly which number maps to which (for inversion warnings in the hybrid functions' execution)
//TODO: (approximator) (setter) clampers for the tunable_parameter setter of the Gaussian approx, and introduce more clamping to approximator setters
//TODO: (base and advanced) (setter) pre-compute minimums and maximums with sub and sum epsilon for readily computed open-interval boundaries

#if !defined(META_LERP) && !defined(LIB_METALERP)
#define META_LERP

    //NOTE: initialize the lib simply with this macro at the very beginning of main
    #ifndef METALERP_INIT
    #define METALERP_INIT metalerp_init();
    #endif

//#define METALERP_INCLUDE_EVERYTHING
#define METALERP_CUDA_LAYER_READY    


#ifdef __cplusplus
    extern "C" {
#endif

    #include "headers/approximator.h" 

#ifdef __cplusplus
    }
#endif

#endif //metalerp C interface

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

#else
//common header has header-guards-independent preprocessor routines to sort out missing utility macros and such
#include "headers/commons.h"

#endif


