/*
proper download of this library from the github repository should give you a copy of the LGPL license text file
that comes with it, if the license was not attained along with this source code; see: (PLACEHOLDER)


This library twists linear interpolation into a fast and powerful
family of transformative mathematical behavior approximators/emulators and constructors
with a very simple interface of at most 2 hyperparameters (for the base forms)
or 4 total hyperparameters for the parametric (in advancedForms.h) forms 
think of it as an implementation of a kind of fast (purely-arithmetic) **non-linear interpolation**.

**************ENTRY HEADER FOR META-LERP**************
you need only to include this header to get the full release
functionality of meta-lerp in C/CPP

you can construct your own new behaviors with visual aid from the desmos 
sheet at: https://www.desmos.com/calculator/cf0389db8e (for messing with parameters and seeing how the functions change) --editor note: link will change most likely


also check out all current approximations I made up until now
at: https://www.desmos.com/calculator/1792cfd969 (you can add more approximators using the lib's functions and parameters yourself in your own local copy,
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


/**************************************************/
/*TODO: 
- GPU layer and runtime handler (switch mechanism)
modification to setters as well (to set both device and host parameters) after making the CUDA kernels (if CUDA is available that is)

- python interface, polished C interface (grouping setters for max and min, and parametric fusing with that too for advanced setters)
to ensure user never makes the mistake of setting min before max (although the bugs emergent from that are handled)

*/

#ifndef META_LERP
#define META_LERP

#define METALERP_INCLUDE_EVERYTHING //includes everything else
#define METALERP_CUDA_LAYER_READY    

#ifdef __cplusplus
    extern "C" {
#endif

    #include "headers/kDispatcher.h" 
    
#ifdef __cplusplus
    }
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
#endif

#endif //metalerp C interface
