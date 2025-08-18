#ifndef BENCHMARKS_H
#define BENCHMARKS_H


/*
TODO: 
do the testing and benchmarking after avdancedForms is done
do the testing again but with GPU kernels

for the future: maybe make a classifying activation (with log maybe)
and test against softmax

*/


#include "../../core_lib/include/headers/commons.h"

#define COS(X) cosf(X)
#define TANH(X) tanhf(X)
#define LN(X) logf(X)
#define LOG(X) log10f(X)

STATIC_FORCE_INLINE type fastSig(type x);
STATIC_FORCE_INLINE type fastSig2(type x);
STATIC_FORCE_INLINE type ReLU(type x);
STATIC_FORCE_INLINE type LeakyReLU(type x);
STATIC_FORCE_INLINE type Sigmoid(type x);
STATIC_FORCE_INLINE type softPlus(type x);
STATIC_FORCE_INLINE type Mish(type x);
STATIC_FORCE_INLINE type Swish(type x);
STATIC_FORCE_INLINE type ReLUReWritten(type x);
STATIC_FORCE_INLINE type Increment(type x);
STATIC_FORCE_INLINE type Addition(type x, type y);
STATIC_FORCE_INLINE type Mul(type x, type y);

STATIC_FORCE_INLINE
type fastSig(type x)
{
    return x / (cast(type, 1) + type_abs(x));
}

STATIC_FORCE_INLINE
type fastSig2(type x)
{
   return (x / (2 * (((x < cast(type, 0)) * (-x)) + ((x >= cast(type, 0)) * x)) + 2) + 0.5f);
}

STATIC_FORCE_INLINE
type ReLU(type x)
{
    return fmaxf(0.f, x);
}
STATIC_FORCE_INLINE
type ReLUReWritten(type x)
{   
    return EXPECT_BRANCH(x > 0) ? x : 0;
}
type alpha = 0.3;
STATIC_FORCE_INLINE
type LeakyReLU(type x)
{
    return ( x > 0 ? x : (alpha * x) );
}

STATIC_FORCE_INLINE
type Sigmoid(type x)
{
    return 1 / (1 + expf(-x)) ;
}

STATIC_FORCE_INLINE
type softPlus(type x)
{
    return logf(1 + expf(x));
}

STATIC_FORCE_INLINE
type Mish(type x)
{
    return x * tanhf(softPlus(x));
}
STATIC_FORCE_INLINE
type Swish(type x)
{
    return x / (1 + expf(-x));
}

STATIC_FORCE_INLINE
type Addition(type x, type y)
{
    return x + y;
}

STATIC_FORCE_INLINE 
type Mul(type x, type y)
{
    return x*y;
}
STATIC_FORCE_INLINE 
type Increment(type x)
{
    return x+1;
}

#endif //BENCHMARKS_H