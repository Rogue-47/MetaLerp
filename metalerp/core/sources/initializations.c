#define INCLUDE_METALERP_INTERNAL_MACRO_UTILS
#define METALERP_CUDA_LAYER_READY

#include "../include/metalerpDefs.h"

//fundamental

BOOL32 metalerpEvenInversionCheckOnce;
BOOL32 metalerpOddInversionCheckOnce;


//OMP specific
uint32_t MP_dispatch_chunksize;
size_t MP_threshold;


BOOL32 metalerp_determineFaster(type* restrict in, type* restrict out, size_t length) //returns 1 if single-threaded is faster, 0 if MP is faster
{   
    METALERP_HEURISTICS_VARIABLES static double ELAPSED_S, ELAPSED_MP;
    
    //takes the brunt of initial page-faults, any work is being done in the 2 loops that access both arrays differently
    for(int64_t i = length-1; i>=0; --i)
    { 
        in[i] = metalerpRand();
        
        out[i] = 0;

    }

    for(size_t i = 0; i<length; ++i)
    {    
        out[i] = in[i] / metalerpRand();
    }

    SLEEP(1);
    
    static size_t prevfunc = cast(size_t, 0);
    size_t randfunc;
    int votes = 0;


    for(int n = 1; n<=METALERP_VOTES_MAX; ++n)
    {
        ELAPSED_S = 0;
        ELAPSED_MP = 0;

        for(size_t a = 0; a < cast(size_t, METALERP_AVERAGING_SIZE); ++a)
        {
            do
            {
                randfunc = cast(size_t, next()%METALERP_ARRSIZE(metalerp_heuristicsOps));
            } while((randfunc == prevfunc));

            prevfunc = randfunc;

            GET_FREQ(&FREQUENCY);
            MEASURE(&START);
            for(size_t i = 0; i<length; ++i)
            {
                out[i] = metalerp_heuristicsOps[randfunc](in[i], in[i]);
            }
            MEASURE(&END);
            ELAPSED_S += calcElapsed(FREQUENCY, START, END);
            SLEEP(1);
            //----------------------------------------
            do
            {
                randfunc = cast(size_t, next()%METALERP_ARRSIZE(metalerp_heuristicsOps));
            } while((randfunc == prevfunc));

            prevfunc = randfunc;
            
            GET_FREQ(&FREQUENCY);
            MEASURE(&START);
            #pragma omp parallel for simd schedule(static, MP_dispatch_chunksize) proc_bind(close) num_threads(omp_get_max_threads())
            for(size_t i = 0; i<length; ++i)
            {
                out[i] = metalerp_heuristicsOps[randfunc](in[i], in[i]);
            }
            MEASURE(&END);
            ELAPSED_MP += calcElapsed(FREQUENCY, START, END);
            SLEEP(1);
        }
        
        ELAPSED_S = ELAPSED_S/cast(double, METALERP_AVERAGING_SIZE);
        ELAPSED_MP = ELAPSED_MP/cast(double, METALERP_AVERAGING_SIZE);

        ELAPSED_S <= ELAPSED_MP ? ++votes : --votes;

        int diff = METALERP_VOTES_MAX - n;

        if( abs(votes) > diff )
            break;

    }

    return (BOOL8)(votes >= 0); //votes can never be zero as long as voting max is odd, so either votes > 0 (single-thread is faster) or votes < 0 (MP is faster)
}

void metalerp_MP_heuristics()
{

    MP_threshold = metalerp_startNum;

    type* in = malloc(MP_threshold * sizeof(type)),
    *out = malloc(MP_threshold * sizeof(type));

    BOOL32 yesBranch = 0, noBranch = 0;

    size_t lower = metalerp_startNum;
    size_t higher = metalerp_startNum;
    
    while(MP_threshold <= metalerp_maxAlloc)
    {
        if(metalerp_determineFaster(in, out, MP_threshold))
        {
            yesBranch += 1;
            lower = MP_threshold;

            if(!noBranch)
            {
                MP_threshold = MP_threshold<<1;
            }
            else
            {
                MP_threshold = (MP_threshold + higher) / 2;
            }
            if((MP_threshold - lower) <= METALERP_THRESHOLD_DIFFERENCE)
                break;
        }
        else
        {
            noBranch += 1;
            higher = MP_threshold;

            if(!yesBranch)
            {
                MP_threshold = MP_threshold>>1;
            }
            else
            {
                MP_threshold = (lower + MP_threshold) / 2;
            }
            if((higher - MP_threshold) <= METALERP_THRESHOLD_DIFFERENCE)
                break;
        }
        in = realloc(in, MP_threshold * sizeof(type));
        out = realloc(out, MP_threshold * sizeof(type));
    }

    free(in); free(out);
    MP_threshold += METALERP_ST_BIAS; //bias toward single-threaded execution more
}

void metalerp_OMP_init()
{
     if(sizeof(type) == 4)
    {
        MP_dispatch_chunksize = 1024;
    }
    else if(sizeof(type) == 8)
    {
        MP_dispatch_chunksize = 512;
    }
    else if(sizeof(type) == 2)
    {
        MP_dispatch_chunksize = 2048;
    }
    else //sizeof(type) == 1, not supported as a storage type size for now though
    {
        MP_dispatch_chunksize = 4096;
    }

    metalerp_MP_heuristics();

}   


void metalerp_init() //only function to call in main
{

    metalerp_CUDA_init();

    selfSeed();

    metalerp_OMP_init();

    metalerpEvenInversionCheckOnce = 0;
    metalerpOddInversionCheckOnce = 0;
}


#undef INCLUDE_METALERP_INTERNAL_MACRO_UTILS