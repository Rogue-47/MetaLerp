#define METALERP_FAST
//plan after this massive bullshit discovery: 
//pull the CUDA layer cleanly off of common header and make a specific header for it goddamnit, this will make integrating



#include "../include/benchmarks.h"

type METALERP_NORMDIST_SQRT;
type BM_std, BM_mean;

void benchmarks_init()
{
    BM_std = 0.5f;
    BM_mean = 1;
    
    METALERP_NORMDIST_SQRT = sqrtf( 2 * cast(type, acosf(0) * 2));

    COPY_PARAM2DEVICE(NM(METALERP_NORMDIST_SQRT), METALERP_NORMDIST_SQRT)
    
    COPY_PARAM2DEVICE(NM(BM_std), BM_std)

    COPY_PARAM2DEVICE(NM(BM_mean), BM_mean)

}