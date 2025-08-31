#ifndef TEST_UTILS_H
#define TEST_UTILS_H
    
    #define INCLUDE_METALERP_INTERNAL_MACRO_UTILS

    #include "../../metalerp.h"

    void setMaxAndMin(type min, type max);
    void setParams(type k, type z);
    void setArms(enum Functions l_Arm, enum Functions r_Arm,
         enum Functions l_Arm_LR, enum Functions_LR r_Arm_LR);

    static const type FLT_EQ_EPS = (type)1e-5;
    #define FLT_EQ(flt1, flt2) tolerantFloatEquality((flt1), (flt2))
    #define FLT_NEQ(flt1, flt2) !FLT_EQ(flt1, flt2)

    BOOL32 tolerantFloatEquality(type flt1, type flt2);
    BOOL32 checkBoundSatisfaction(type in, type min, type max); //check if input satisfies the range that is given



#endif //header