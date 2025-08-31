#include "../include/test_utils.h"


void setMaxAndMin(type min, type max) //for debugging purposes but it simplifies hyperparam config through the higher-level interface too
{

    setMinA_E(min);
    setMinA_O(min);
    
    setMaxA_E(max);
    setMaxA_O(max);

    setMaxD_E(max);
    setMaxD_O(max);

    setMinD_E(min);
    setMinD_O(min);

    p_setMinA_E(min);
    p_setMinA_O(min);

    p_setMaxA_E(max);
    p_setMaxA_O(max);

    p_setMaxD_E(max);
    p_setMaxD_O(max);

    p_setMinD_E(min);
    p_setMinD_O(min);
}
void setParams(type k, type z)
{
    p_setZ_A_O(z);
    p_setZ_A_E(z);
    p_setZ_D_E(z);
    p_setZ_D_O(z);

    p_setK_A_O(k);
    p_setK_A_E(k);
    p_setK_D_E(k);
    p_setK_D_O(k);
}
void setArms(enum Functions l_Arm, enum Functions r_Arm, enum Functions l_Arm_LR, enum Functions_LR r_Arm_LR)
{
    setHybridComboArms(l_Arm, r_Arm);
    setHybridComboArms_LR(l_Arm_LR, r_Arm_LR);
}

BOOL32 tolerantFloatEquality(type flt1, type flt2)
{
    return flt1 != flt2 ? (type_abs(flt1 - flt2) <= FLT_EQ_EPS) : 1;
}

BOOL32 checkBoundSatisfaction(type in, type min, type max) //closed interval implementation by default
{
    if((in < min) || (in > max))
        return 0;
    return 1;
}

//still need immunity checks to domain restriction correct side-effects (to ignore them)
//and also arm-biasing dependent output (to filter that out too)