# MetaLerp

## metalerp is a lightweight, and ***fast*** mathematical behavior construction and emulation library for non-linear functions, it integrates natively into C/C++ or can be linked dynamically for other workspaces, and has a python-packaged interface.

### Both interfaces have GPU execution on machines with nvidia cards 

## Why would you use it?  

### The library provides configurable formulas with bounding behavior on any real input, small variations in some of the formulas of the library produced accurate and noticeably faster approximations to some of the most popular transformations in numeric compute.

 - ### Dynamic Simulations, Graphics transformations and Shaders, Animation - if it needs standard Lerp - then it might prove interesting to see the behavior of the non-linear interpolation formulas provided here.

 - ### Machine Learning (as activation or normalization/standardization functions). 

 - ### Experimentation in Cryptographic Algorithms: the oddly symmetric transformations' behavior has proven to have an overlapping inverse domain when the minimum hyperparameter (a) < 0, this means that odd formulas' inverse domains have to be limited to make sense as they are not invertible. I believe this provides some basic cryptographic strength and a building block in crypt algorithms.

 - ### Utilities: maybe you need to make a Gaussian PRNG? you could rely on the fast approximation to the Gaussian provided by metalerp, or you need just a bounding formula that follows certain behaviors and is configurable with atleast 2 parameters? you could use metalerp's base or parametric transforms

## Generally it can be applied in any domain where non-linear transformation of something to an unbounded or bounded range in many different ways, or vice-versa (bounded to unbounded) by using the inverse tranformations.

# Desmos Sheets:
### The following sheets are helpful for visualization and seeing the effect of configuring the parameters of the formulas
## [Demonstration](https://www.desmos.com/calculator/cf0389db8e)
## [Approximator Demonstration](https://www.desmos.com/calculator/neji45kf1n)

# How to use (C/C++):
### Building 
### To see the function interface of the lib, simply look in the metalerp/metalerp.h file - it defines all the functions you can directly use.

## **it is recommended that you type METALERP_INIT at the beginning inside of the main function, it is a macro that calls the lib's full initialization routine**

### metalerp's Makefile provides an easy way to automate building the lib's binaries and even link them with your main program

    git clone https://github.com/Rogue-47/MetaLerp.git
    cd MetaLerp

### For standard assertion builds with regular 32-bit float as the native compute type:

#### shared lib build, **include "metalerp/metalerp.h"** in the main file and any other source file that requires metalerp kernels:
    make f32 && make

#### in the source file (if it is inside MetaLerp directory):
    #include "metalerp/metalerp.h"
    //....
    int main()
    {
        METALERP_INIT
        //....
        return 0;
    }

### The first make command compiles the shared library, and the second links it to your main file if everything was successful - by default the second make command assumes there is a main.c file present in the directory, and outputs by default a program called "main" if everything was successful

#### static lib build, same as shared build; **include: "metalerp/metalerp.h"**:
    make static_f32 && make

#### development build, **include: "metalerp/core/include/metalerpDefs.h"** or **"metalerp/auxiliary/metalerpTest.h"** (monolothic compilation, compile the lib files and main program all in one go):
    make dev_f32

#### in the source file (if it is inside MetaLerp directory):
    #include "metalerp/core/include/metalerpDefs.h"
    //or: #include "metalerp/auxiliary/metalerpTest.h" (for test utilities, and perf measurement functionality tailored for metalerp)
    //....
    int main()
    {
        METALERP_INIT
        //....
        return 0;
    }

### it is not needed to call the second make command since dev targets do all the linking with the main program already in the first make command - you only need to call the first make command and you'll have your program ready.

### The makefile has **f64** targets too, and release and fast versions of the lib, release does if checks in the batched kernel dispatchers and CUDA routine calls without assertions, and fast build does not check inside the dispatchers at all (checks are for input and output buffer length matches, input and output buffer validity, etc.)

### for just generating the shared/static library, you can rely on the make f32/f64 commands with the proper prefix for your desired binary object, but be careful as specifying anything other than the default f32 or static_f32 targets introduces macro definitions that compile a certain configuration of the lib... if neither the source file using this lib nor its compilation command have those macros defined - it easily causes ABI mismatch and runtime bugs will occur, but you can easily know which macros were defined by calling:
    cat bin/*
### inside the same directory as the makefile, this will output the content of the shell file that compiles it with the main program, where you can see the -D macro defines necessary to have for any source file using that specific configuration of the lib's binary

# For the python package:
    pip install metalerp
### and just import it in any .py file Note: **the package is available as a source dist, meaning you'll need NVIDIA's nvcc compiler (with a CUDA_PATH environment variable set, pointing to its folder that is the parent of the bin/ folder), and a host C Compiler (currently, only gcc supported) to build the package.**

# Minimal example (lib builds):
### C/C++:
    //main.c
    #define INCLUDE_METALERP_INTERNAL_MACRO_UTILS /*to get the type macro and not have to refactor the data type just in case you switch between float (f32) and double (f64) builds a lot*/
    #include <time.h>    
    #include <stdlib.h>
    #include <stdio.h>
    #include "metalerp/metalerp.h"
    int main()
    {
        srand(time(NULL));
        size_t asize = 1<<20;

        METALERP_INIT
        
        type input = malloc(sizeof(type)*asize);
        type output = malloc(sizeof(type)*asize);
        
        for(size_t i = 0; i<asize; ++i) input[i] = (type)rand();
        
        /*
        B_A_E = Base Ascending (as in, as x increases, y increases until it hits the boundary of the transformation) Even (evenly symmetric) (transform), in other dispatchers, D stands for descending and O stands for Odd, inv_ prefixes indicate that it is an inverse of the forward function, meaning batched_inv_B_A_E is the inverse (not perfectly, since it is an evenly symmetric function) of B_A_E
        */
        batched_B_A_E(input, output, asize, asize);
        //if lib build was a fast build, use: batched_B_A_E(input, output, asize);
        
        for(size_t i = 0; i<asize; i+= ((asize)/(1<<18))) printf("x = %f, y = %f\n", input[i], output[i]);
        free(input); free(output);
        return 0;
    }
### Python:
    //main.py
    import metalerp as ml
    import numpy as np
    
    N = int(1e+6)

    data = np.ones(N, dtype=ml.metalerp_dtype)

    """
    data with dtype of metalerp_dtype help speed up metalerp methods since
    the method won't have to make a conversion on the data
    specifically to metalerp's native data type (which always happens)
    to compute the output.

    The compute methods accept a single scalar number, just about any generic iterable (except for jagged lists, or dictionaries), and numpy arrays, all of any number data type.
    """
    print(ml.B_A_E(data))


# Mathematical Behaviors:  
### input: x $\in$ R, output: y $\in$ [a, b] if the transform is an ascending variant, or y $\in$ [b, a] if the transform is a descending variant, with parametric variants introducing the K and Z (called V in the approximator functions) to modulate the sharpness of the curve and flip/attenuate its peak respectively, without loss of compute speed. There's also a hybrid function which can be used to combine different transform behaviors; with it you can assign a transform for x<0, and a different one for x>0.

### Proofs have been done for the formulas to prove boundary compliances at extreme values and at zero, and will be uploaded soon  
### the fixed behavior for the transforms is:  
### At x=0, the transform yields its minimum value (parameter a) if it's an ascending variant, or yields its maximum (parameter b) if it is descending
### The more b -> $\infty$, the more accurate the expression ( y $\approx$ x + a) becomes, for the ascending transform; and vice-versa for the descending


## Collaboration in expanding, refining, polishing, and/or demonstrating/making more examples (approximation and emulation functions) of this library and its mathematical foundation is welcome. If you messed with the formulas of this lib and managed to come up with anything cool, please publish



## Disclaimer Section:
### Metalerp started as and continues to be an experimental hobby project as of now, if any bugs are found in the lib's runtime or with the lib's compilation interface (especially if you experimented with further configuration of the lib's behavior via the common header), please contact me at metalerplib@gmail.com.

### ***I guarantee no warranty or liability to the usage and deployment of this library in software, per the Lesser GNU Public License.***

## - Rogue-47 / Omar M. Mahmoud