#include "../../include/headers/external/xoshiro256plus.h"
#include "../../include/headers/external/xoshiro256plusplus.h"

//MetaLerp's main randomization utilities
//slight additions and even tinier modifications made to the xoshiro256+ and ++ functionality
//implemented originally at: https://prng.di.unimi.it
/*********externals**********/

//xoshiro256+
uint64_t XOSHIRO256PLUS_STATE[4];

uint64_t nextPlus(void) {
	const uint64_t result = XOSHIRO256PLUS_STATE[0] + XOSHIRO256PLUS_STATE[3];

	const uint64_t t = XOSHIRO256PLUS_STATE[1] << 17;

	XOSHIRO256PLUS_STATE[2] ^= XOSHIRO256PLUS_STATE[0];
	XOSHIRO256PLUS_STATE[3] ^= XOSHIRO256PLUS_STATE[1];
	XOSHIRO256PLUS_STATE[1] ^= XOSHIRO256PLUS_STATE[2];
	XOSHIRO256PLUS_STATE[0] ^= XOSHIRO256PLUS_STATE[3];

	XOSHIRO256PLUS_STATE[2] ^= t;

	XOSHIRO256PLUS_STATE[3] = rotlPlus(XOSHIRO256PLUS_STATE[3], 45);

	return result;
}

float nextF(void) {
	const uint64_t result = XOSHIRO256PLUS_STATE[0] + XOSHIRO256PLUS_STATE[3];

	const uint64_t t = XOSHIRO256PLUS_STATE[1] << 17;

	XOSHIRO256PLUS_STATE[2] ^= XOSHIRO256PLUS_STATE[0];
	XOSHIRO256PLUS_STATE[3] ^= XOSHIRO256PLUS_STATE[1];
	XOSHIRO256PLUS_STATE[1] ^= XOSHIRO256PLUS_STATE[2];
	XOSHIRO256PLUS_STATE[0] ^= XOSHIRO256PLUS_STATE[3];

	XOSHIRO256PLUS_STATE[2] ^= t;

	XOSHIRO256PLUS_STATE[3] = rotlPlus(XOSHIRO256PLUS_STATE[3], 45);

	return ((float) (result>>40) )/( (float) (1ULL<<24) ); //extract the higher bits with a normalized fp32 mantissa, since the suggestion is to avoid the lower bits that aren't as entropic.
}

void selfSeedPlus() //usable at init or runtime
{
    static int firstInitP = 1;

    if(firstInitP)
    {
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC, &ts); //high res
    srand((unsigned int)(ts.tv_nsec ^ time(NULL)));
    firstInitP = 0;
    XOSHIRO256PLUS_STATE[0] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand())); /*since rand is a 16-bit weakling*/
    XOSHIRO256PLUS_STATE[1] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    XOSHIRO256PLUS_STATE[2] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    XOSHIRO256PLUS_STATE[3] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    }
    
    //n = 3 iterations is just an arbitrary number here.
    for(int i = 0; i<3; ++i)
    {
        XOSHIRO256PLUS_STATE[0] = nextPlus();
        XOSHIRO256PLUS_STATE[1] = nextPlus();
        XOSHIRO256PLUS_STATE[2] = nextPlus();
        XOSHIRO256PLUS_STATE[3] = nextPlus();
    }
}

//**************************************xoshiro256++
uint64_t XOSHIRO256PLUSPLUS_STATE[4];
uint64_t next(void) {
	const uint64_t result = rotl(XOSHIRO256PLUSPLUS_STATE[0] + XOSHIRO256PLUSPLUS_STATE[3], 23) + XOSHIRO256PLUSPLUS_STATE[0];

	const uint64_t t = XOSHIRO256PLUSPLUS_STATE[1] << 17;

	XOSHIRO256PLUSPLUS_STATE[2] ^= XOSHIRO256PLUSPLUS_STATE[0];
	XOSHIRO256PLUSPLUS_STATE[3] ^= XOSHIRO256PLUSPLUS_STATE[1];
	XOSHIRO256PLUSPLUS_STATE[1] ^= XOSHIRO256PLUSPLUS_STATE[2];
	XOSHIRO256PLUSPLUS_STATE[0] ^= XOSHIRO256PLUSPLUS_STATE[3];

	XOSHIRO256PLUSPLUS_STATE[2] ^= t;

	XOSHIRO256PLUSPLUS_STATE[3] = rotl(XOSHIRO256PLUSPLUS_STATE[3], 45);

	return result;
}
void selfSeed() //usable at init or runtime
{  
	static int firstInit = 1;

    if(firstInit)
    {
	struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC, &ts); //high res
    srand((unsigned int)(ts.tv_nsec ^ time(NULL)));
    firstInit = 0;
    XOSHIRO256PLUSPLUS_STATE[0] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    XOSHIRO256PLUSPLUS_STATE[1] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    XOSHIRO256PLUSPLUS_STATE[2] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    XOSHIRO256PLUSPLUS_STATE[3] = ( ((uint64_t)rand()) << 48) ^ ( ((uint64_t)rand()) << 32) ^ ( ((uint64_t)rand()) << 16) ^ (((uint64_t)rand()));
    }
	
    //n = 3 iterations is just an arbitrary number here.
    for(int i = 0; i<3; ++i)
    {
        XOSHIRO256PLUSPLUS_STATE[0] = next();
        XOSHIRO256PLUSPLUS_STATE[1] = next();
        XOSHIRO256PLUSPLUS_STATE[2] = next();
        XOSHIRO256PLUSPLUS_STATE[3] = next();
    }

}