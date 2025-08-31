//standard forms of the transformation formula: 

#ifndef BASE_FORMS
#define BASE_FORMS

#include "commons.h"

//this header should only be included individually in combos.h for proper functionality


        #ifdef ODD_EVEN_SAME_HYPERPARAMS
    
        //for the ascender
        extern type minA;
        extern type maxA;
        //for descender
        extern type minD;
        extern type maxD;
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
        //for the ascender
        extern type minA_E;
        extern type maxA_E;
    
        extern type minA_O;
        extern type maxA_O;
        //for descender
        extern type minD_E;
        extern type maxD_E;
    
        extern type minD_O;
        extern type maxD_O;
    
        #define min_A_E minA_E
        #define max_A_E maxA_E
        #define min_A_O minA_O
        #define max_A_O maxA_O
        /***********/
        #define min_D_E minD_E
        #define max_D_E maxD_E
        #define min_D_O minD_O
        #define max_D_O maxD_O
    
        #endif //hyperparam resolution

#ifdef BRANCHING_SETTERS
        #ifdef ODD_EVEN_SAME_HYPERPARAMS

            void setMin(type min, BOOL32 Flag);

            void setMax(type max, BOOL32 Flag);
        #else

            void setMin_E(type min, BOOL32 Flag);

            void setMax_E(type max, BOOL32 Flag);

            void setMin_O(type min, BOOL32 Flag);

            void setMax_O(type max, BOOL32 Flag);
        #endif
#else //faster setters (non-branching), default
    /*minimizes the overhead of having to pass the extra parameter to specify which base variant to change the global parameters of AND to skip the branching
    essentially making the idea of using them as learnabe parameters (that're adjusted frequently) more palpable*/
        
        #ifdef ODD_EVEN_SAME_HYPERPARAMS
        

            void setMinA(type min);

            void setMaxA(type max);

            void setMinD(type min);

            void setMaxD(type max);
        #else
            

            void setMinA_E(type min);

            void setMinA_O(type min);
        

            void setMaxA_E(type max);

            void setMaxA_O(type max);
        

            void setMinD_E(type min);

            void setMinD_O(type min);
        

            void setMaxD_E(type max);

            void setMaxD_O(type max);
            
        #endif
#endif //BRANCHING_SETTERS

#ifdef __CUDACC__

    //for the ascender
    __device__ type D_bParams[8];

    #define D_min_A_E D_bParams[0]
    #define D_max_A_E D_bParams[1]
    
    #define D_min_A_O D_bParams[2]
    #define D_max_A_O D_bParams[3]
    /***********/
    #define D_min_D_E D_bParams[4]
    #define D_max_D_E D_bParams[5]
    
    #define D_min_D_O D_bParams[6]
    #define D_max_D_O D_bParams[7]
    
    #define D_min_A_E_idx 0
    #define D_max_A_E_idx 1
    
    #define D_min_A_O_idx 2
    #define D_max_A_O_idx 3
    /***********/
    #define D_min_D_E_idx 4
    #define D_max_D_E_idx 5
    
    #define D_min_D_O_idx 6
    #define D_max_D_O_idx 7

    #ifdef BRANCHING_SETTERS
        #ifdef ODD_EVEN_SAME_HYPERPARAMS
    
            //Flag: 1 or greater changes ascending variant's 
            //parameters, 0 changes descending variant's parameters
METALERP_DEF_SHIM_2ARG(setMin, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) { min_D_E = SET_MIN(min, max_D_E); COPY_PARAM2DEVICE(D_bParams, min_D_E)}
                else { min_A_E = SET_MIN(min, max_A_E); COPY_PARAM2DEVICE(D_bParams, min_A_E)} 
            }
METALERP_DEF_SHIM_2ARG(setMax, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) {max_D_E = SET_MAX(max); ENFORCE_MAX(min_D_E, max_D_E); COPY_PARAM2DEVICE(D_bParams, max_D_E)}
                else {max_A_E = SET_MAX(max); ENFORCE_MAX(min_A_E, max_A_E); COPY_PARAM2DEVICE(D_bParams, max_A_E)}
            }
        #else
        
            //Flag: 1 or greater changes ascending variant's 
            //parameters, 0 changes descending variant's parameters
METALERP_DEF_SHIM_2ARG(setMin_E, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) { min_D_E = SET_MIN(min, max_D_E); COPY_PARAM2DEVICE(D_bParams, min_D_E)}
                else { min_A_E = SET_MIN(min, max_A_E); COPY_PARAM2DEVICE(D_bParams, min_A_E)}
            }
METALERP_DEF_SHIM_2ARG(setMax_E, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) { max_D_E = SET_MAX(max); ENFORCE_MAX(min_D_E, max_D_E); COPY_PARAM2DEVICE(D_bParams, max_D_E)}
                else { max_A_E = SET_MAX(max); ENFORCE_MAX(min_A_E, max_A_E); COPY_PARAM2DEVICE(D_bParams, max_A_E)}
            }
METALERP_DEF_SHIM_2ARG(setMin_O, type, min, BOOL32, Flag)(type min, BOOL32 Flag)
            {
                if(Flag == 0) { min_D_O = SET_MIN(min, max_D_O); COPY_PARAM2DEVICE(D_bParams, min_D_O)}
                else { min_A_O = SET_MIN(min, max_A_O); COPY_PARAM2DEVICE(D_bParams, min_A_O)}
            }
METALERP_DEF_SHIM_2ARG(setMax_O, type, max, BOOL32, Flag)(type max, BOOL32 Flag)
            {
                if(Flag == 0) { max_D_O = SET_MAX(max); ENFORCE_MAX(min_D_O, max_D_O); COPY_PARAM2DEVICE(D_bParams, max_D_O)}
                else { max_A_O = SET_MAX(max); ENFORCE_MAX(min_A_O, max_A_O); COPY_PARAM2DEVICE(D_bParams, max_A_O)}
            }
        #endif
    #else //faster setters (non-branching), default
    /*minimizes the overhead of having to pass the extra parameter to specify which base variant to change the global parameters of AND to skip the branching
    essentially making the idea of using them as learnabe parameters (that're adjusted frequently) more palpable*/
        
        #ifdef ODD_EVEN_SAME_HYPERPARAMS
        
METALERP_DEF_SHIM(setMinA, type, min)(type min)
            {
                min_A_E = SET_MIN(min, max_A_E); COPY_PARAM2DEVICE(D_bParams, min_A_E)
            }
METALERP_DEF_SHIM(setMaxA, type, max)(type max) 
            {
                max_A_E = SET_MAX(max); ENFORCE_MAX(min_A_E, max_A_E); COPY_PARAM2DEVICE(D_bParams, max_A_E)
            }
METALERP_DEF_SHIM(setMinD, type, min)(type min)
            {
                min_D_E = SET_MIN(min, max_D_E); COPY_PARAM2DEVICE(D_bParams, min_D_E)
            }
METALERP_DEF_SHIM(setMaxD, type, max)(type max)
            {
                max_D_E = SET_MAX(max); ENFORCE_MAX(min_D_E, max_D_E); COPY_PARAM2DEVICE(D_bParams, max_D_E)
            }
        #else
            
METALERP_DEF_SHIM(setMinA_E, type, min)(type min)
            {
                min_A_E = SET_MIN(min, max_A_E);
                COPY_PARAM2DEVICE(D_bParams, min_A_E);
            }
METALERP_DEF_SHIM(setMinA_O, type, min)(type min) { min_A_O = SET_MIN(min, max_A_O); COPY_PARAM2DEVICE(D_bParams, min_A_O)}
        
METALERP_DEF_SHIM(setMaxA_E, type, max)(type max) { max_A_E = SET_MAX(max); ENFORCE_MAX(min_A_E, max_A_E); COPY_PARAM2DEVICE(D_bParams, max_A_E)}
METALERP_DEF_SHIM(setMaxA_O, type, max)(type max) { max_A_O = SET_MAX(max); ENFORCE_MAX(min_A_O, max_A_O); COPY_PARAM2DEVICE(D_bParams, max_A_O)}

METALERP_DEF_SHIM(setMinD_E, type, min)(type min) { min_D_E = SET_MIN(min, max_D_E); COPY_PARAM2DEVICE(D_bParams, min_D_E)}
METALERP_DEF_SHIM(setMinD_O, type, min)(type min) { min_D_O = SET_MIN(min, max_D_O); COPY_PARAM2DEVICE(D_bParams, min_D_O)}
        
METALERP_DEF_SHIM(setMaxD_E, type, max)(type max) { max_D_E = SET_MAX(max); ENFORCE_MAX(min_D_E, max_D_E); COPY_PARAM2DEVICE(D_bParams, max_D_E)}
METALERP_DEF_SHIM(setMaxD_O, type, max)(type max) { max_D_O = SET_MAX(max); ENFORCE_MAX(min_D_O, max_D_O); COPY_PARAM2DEVICE(D_bParams, max_D_O)}
        #endif
    #endif //BRANCHING_SETTERS

#endif //CUDA LAYER

#if defined(METALERP_REDEF_LATER) && defined (INSIDE_COMBOS_H)
    
    #undef min_A_E
    #undef max_A_E
    #undef min_A_O
    #undef max_A_O

    #undef min_D_E
    #undef max_D_E
    #undef min_D_O
    #undef max_D_O

    #define min_A_E NM(min_A_E)
    #define max_A_E NM(max_A_E)
    #define min_A_O NM(min_A_O)
    #define max_A_O NM(max_A_O)
    /***********/
    #define min_D_E NM(min_D_E)
    #define max_D_E NM(max_D_E)
    #define min_D_O NM(min_D_O)
    #define max_D_O NM(max_D_O)

#endif

/*the basic bounding forms*/
//base variant and signed additions (even symmetric (_e) transformation functions and odd-symmetric (_o) forms with the sgn calls (a little heavier))
/* 
    form1: (ascender)
    b * (a + |x|)
y = _________________  -> (|x|/(b+|x|) * (b-a)+a
     (b + |x| )

     x->{a, b} or {-b, -a} with sgn functionality

    form2: (b/(b+|x|)) * (b-a)+a (descender)
    b^2 + (a*|x|)
y = _________________
     (b + |x| )
     
     x->{b, a} or {-a, -b} with sgn functionality
*/

METALERP_INTERNAL_DEVFUNC
type NM(ascendingVariant_E)(type x);
METALERP_INTERNAL_DEVFUNC
type NM(descendingVariant_O)(type x);

METALERP_INTERNAL_DEVFUNC
type NM(descendingVariant_E)(type x); 

METALERP_INTERNAL_DEVFUNC
type NM(ascendingVariant_O)(type x);

//for the combo merger mainly
METALERP_INTERNAL_DEVFUNC
type NM(ascendingV_O_RightArm)(type x);
METALERP_INTERNAL_DEVFUNC
type NM(ascendingV_O_LeftArm)(type x);

METALERP_INTERNAL_DEVFUNC
type NM(descendingV_O_RightArm)(type x);
METALERP_INTERNAL_DEVFUNC
type NM(descendingV_O_LeftArm)(type x);

/********************************************************/
/*inverses of the bounded forms for non-bounded behavior (exponential-like or factorial-like etc.)*/

/*
(all constraint issues (e.g. y!=b or y!=a) automatically solved with epsilon shifts in the clamp functions)

    form1: (ascender inverse) (even or positive right arm with + only (even variant inverses naturally produce ambiguously-signed results))
        b * (y - a)
x = ± _________________  -> y must stay in (a, b), y != b
         (b - y)


    form2: (descender) (even or positive right arm with + only)
         b * (y - b)
x = ±  _________________ y must stay in (a, b) and a != 0, y != a
            (a - y)
     
*/
//clamp functions to enforce bounded output domain for the functions
METALERP_INTERNAL_DEVFUNC
type NM(clampY_A_Even)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(clampY_D_Even)(type y);

METALERP_INTERNAL_DEVFUNC
type NM(clampY_A_R)(type y); //right-arm and left-arm specific clampers are for the odd-symmetry functions
METALERP_INTERNAL_DEVFUNC
type NM(clampY_A_L)(type y);

METALERP_INTERNAL_DEVFUNC
type NM(clampY_D_R)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(clampY_D_L)(type y);


METALERP_INTERNAL_DEVFUNC
type NM(inv_ascendingVariant_E)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(inv_descendingVariant_E)(type y); 


METALERP_INTERNAL_DEVFUNC
type NM(inv_ascendingVariant_O)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(inv_ascendingV_O_RightArm)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(inv_ascendingV_O_LeftArm)(type y);

METALERP_INTERNAL_DEVFUNC
type NM(inv_descendingVariant_O)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(inv_descendingV_O_RightArm)(type y);
METALERP_INTERNAL_DEVFUNC
type NM(inv_descendingV_O_LeftArm)(type y);

/*implementations*/

METALERP_INTERNAL_KERNEL(ascendingVariant_E)(type x)
{
    #ifdef NO_ABS
    /*preferably in a less safe setting where user is sure 
    no negative values of x will be passed to the normalization function
    because un-regulated negative inputs will very easily break the even functions
    so abs is done on the input, and output is not affected by sign of input
    NOTE: odd functions handle negatives and positives contextually regarding sign factors
    if specifically x is equal to the max (division by zero) 
    or if it's negative is less than -max 
    (negation of the formula - unexpected behavior)*/
        #define absVar_A x
    #else
        type absVar_A = type_abs(x);
    #endif

    type t = absVar_A/(absVar_A + max_A_E);
       
    return LERP(min_A_E, max_A_E, t);
}
METALERP_INTERNAL_KERNEL(ascendingVariant_O)(type x)
{
    #ifndef NO_ABS
        type absVar_A = type_abs(x);
    #endif
    
    type t = absVar_A/(absVar_A + max_A_O);
    
    return LERP(min_A_O, max_A_O, t) * BSGN(x);
}

METALERP_INTERNAL_KERNEL(descendingVariant_E)(type x) 
{
    #ifdef NO_ABS
        #define absVar_D x
    #else
        #define absVar_D type_abs(x)
    #endif

    type t = max_D_E/(absVar_D + max_D_E);
    
    return LERP(min_D_E, max_D_E, t);
}
METALERP_INTERNAL_KERNEL(descendingVariant_O)(type x)
{
    type t = max_D_O/(absVar_D + max_D_O);
    
    return LERP(min_D_O, max_D_O, t) * BSGN(x);
}
/*
METALERP_INTERNAL_KERNEL(ascendingVariant_O2)(type x) //second version of the ascending odd variant, appears to be perform slightly better in multi-processed batch versions but slightly worse in everything else?
{
    return(x ASC_POS_COMPARISON cast(type, 0)) ?
     NM(ascendingV_O_RightArm)(x) : NM(ascendingV_O_LeftArm)(x);
}

METALERP_INTERNAL_KERNEL(descendingVariant_O2)(type x)
{
    return( EXPECT_BRANCH(x DESC_POS_COMPARISON cast(type, 0)) ?
     NM(descendingV_O_RightArm)(x) : NM(descendingV_O_LeftArm)(x) );
}
*/


METALERP_INTERNAL_DEVFUNC type
NM(ascendingV_O_RightArm)(type x)
{
    type t = x / (x + max_A_O);
    return LERP(min_A_O, max_A_O, t);
}
METALERP_INTERNAL_DEVFUNC type
NM(ascendingV_O_LeftArm)(type x)
{
    type t = x / (max_A_O - x);
    return minusMinLERP(min_A_O, max_A_O, t);
}
METALERP_INTERNAL_DEVFUNC type
NM(descendingV_O_RightArm)(type x)
{
    type t = max_D_O / (x + max_D_O);
    return LERP(min_D_O, max_D_O, t);
}
METALERP_INTERNAL_DEVFUNC type
NM(descendingV_O_LeftArm)(type x)
{
    type t = max_D_O / (x - max_D_O);
    return minusMinLERP(min_D_O, max_D_O, t);
}

/*********************************/
//TODO: maybe another micro-optimization, pre-compute minimums and maximums with sub and sum epsilon for readily computed open-interval boundaries
METALERP_INTERNAL_DEVFUNC type NM(clampY_A_Even)(type y)
{   //stay inside (a, b) open interval, because equalling either one of those parameters can result in division by zero in the inverse formulas (both ascender and descender can suffer from this)
    //the value will certainly be comically large with such a small denominator that results from subtractions like: 5 - 4.9... but atleast it won't be a zero in the denom
    return type_min(type_max(y, min_A_E SUM_EPSILON(min_A_E)), max_A_E SUB_EPSILON(max_A_E));
}
METALERP_INTERNAL_DEVFUNC type NM(clampY_D_Even)(type y)
{
    return type_min(type_max(y, min_D_E SUM_EPSILON(min_D_E)), max_D_E SUB_EPSILON(max_D_E));
}
/*
METALERP_INTERNAL_DEVFUNC type NM(clampY_A_Odd)(type y, type sgny)
{
    return sgny * (type_min(type_max(sgny * y, min_A_O SUM_EPSILON(min_A_O)), max_A_O SUB_EPSILON(max_A_O)));
}
    */
/*
METALERP_INTERNAL_DEVFUNC type NM(clampY_D_Odd)(type y, type sgny)
{
    return sgny * (type_min(type_max(sgny * y, min_D_O SUM_EPSILON(min_D_O)), max_D_O SUB_EPSILON(max_D_O)));
}
*/
METALERP_INTERNAL_DEVFUNC type NM(clampY_A_R)(type y)
{
    if( min_A_O ASC_POS_COMPARISON cast(type, 0) )
        return type_min(type_max(y, OPEN_INTERVAL_MIN(min_A_O)),
        OPEN_INTERVAL_MAX(max_A_O));
    return type_min(type_max(-y, OPEN_INTERVAL_MIN(min_A_O) ), OPEN_INTERVAL_MAX(cast(type, 0)));
}
METALERP_INTERNAL_DEVFUNC type NM(clampY_A_L)(type y)
{
    if( min_A_O ASC_POS_COMPARISON cast(type, 0) )
        return type_min(type_max(y, -OPEN_INTERVAL_MIN(max_A_O) ), type_min(-OPEN_INTERVAL_MAX(min_A_O), cast(type, 0) ));
    return type_min(type_max(-y, OPEN_INTERVAL_MIN(cast(type, 0)) ), -OPEN_INTERVAL_MAX(min_A_O));
}

METALERP_INTERNAL_DEVFUNC type NM(clampY_D_R)(type y)
{
    return type_min(type_max(y, type_max(cast(type, 0), OPEN_INTERVAL_MIN(min_D_O)) ), OPEN_INTERVAL_MAX(max_D_O));
}
METALERP_INTERNAL_DEVFUNC type NM(clampY_D_L)(type y)
{
    return type_min(type_max(y, -OPEN_INTERVAL_MIN(max_D_O)), type_min(cast(type, 0), -OPEN_INTERVAL_MAX(min_D_O)));
}
/*********************************/
/*
METALERP_INTERNAL_KERNEL
type altClampY_A_Even(type y)
{
    if(y <= min_A_E)
        return min_A_E SUM_EPSILON(min_A_E);
    else if(y >= max_A_E)
        return max_A_E SUB_EPSILON(max_A_E);
    else
        return y;
}
*/
METALERP_INTERNAL_KERNEL(inv_ascendingVariant_E)(type y)
{
    type yClamped = NM(clampY_A_Even)(y);
    return type_fma(max_A_E, yClamped, -(max_A_E * min_A_E)) / (max_A_E - yClamped);
}

METALERP_INTERNAL_KERNEL(inv_descendingVariant_E)(type y)
{
    type yClamped = NM(clampY_D_Even)(y);
    return type_fma(max_D_E, yClamped, -(max_D_E * max_D_E)) / (min_D_E - yClamped);
}

/*
METALERP_INTERNAL_KERNEL
type inv_ascendingVariant_O2(type y)
{
    return EXPECT_BRANCH(y ASC_POS_COMPARISON 0) ? inv_ascendingV_O_RightArm(y) : inv_ascendingV_O_LeftArm(y) ;
}
*/

//TODO (later stages, perhaps post-release): some optimizations here would be nice, this function's branches are few but each one very heavy (bad CPU pipeline stalling).
//possible alternative to the current odd variant scheme: compute sign of y and based on that make a fused left and right arm version that utilizes the sign of y
//leaving the only branching part to be the clamp arm calls, which may be heavier in totality but less affected by branch miss penalties
METALERP_INTERNAL_KERNEL(inv_ascendingVariant_O)(type y)
{
    return NBSGN(min_A_O) * ((y ASC_POS_COMPARISON cast(type, 0)) ? NM(inv_ascendingV_O_RightArm)(y) :
    NM(inv_ascendingV_O_LeftArm)(y));
}

METALERP_INTERNAL_DEVFUNC type NM(inv_ascendingV_O_RightArm)(type y)
{
    type yClamped = NM(clampY_A_R)(y);
    //printf("yClamped retrieved(RA): %f\n", yClamped);
    return type_fma(max_A_O, yClamped, -(max_A_O * min_A_O)) / (max_A_O - yClamped);
}

METALERP_INTERNAL_DEVFUNC type NM(inv_ascendingV_O_LeftArm)(type y)
{
    type yClamped = NM(clampY_A_L)(y);
    //printf("yClamped retrieved(LA): %f\n", yClamped);
    return type_fma(max_A_O, yClamped, max_A_O * min_A_O) / (max_A_O + yClamped);
}

METALERP_INTERNAL_KERNEL(inv_descendingVariant_O)(type y)
{
    return (y DESC_POS_COMPARISON cast(type, 0)) ? NM(inv_descendingV_O_RightArm)(y) :
    NM(inv_descendingV_O_LeftArm)(y);
}

/*
METALERP_INTERNAL_KERNEL(inv_descendingVariant_O2)(type y)
{
    type sgny = NBSGN(y);
    type yClamped = clampY_D_Odd(y, sgny);
   type max = sgny*max_D_O; 
   return type_fma(max, max, -(max * yClamped)) / (yClamped - (sgny * min_D_O));
}
*/

METALERP_INTERNAL_DEVFUNC type NM(inv_descendingV_O_RightArm)(type y)
{
    type yClamped = NM(clampY_D_R)(y);
    return type_fma(max_D_O, yClamped, -(max_D_O * max_D_O)) / (min_D_O - yClamped);
}
METALERP_INTERNAL_DEVFUNC type NM(inv_descendingV_O_LeftArm)(type y)
{
    type yClamped = NM(clampY_D_L)(y);
    return type_fma(max_D_O, yClamped, max_D_O * max_D_O) / (min_D_O + yClamped);
}



#endif //HEADER