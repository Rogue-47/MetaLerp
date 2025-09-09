/*Metalerp Python Interface*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#define INCLUDE_METALERP_INTERNAL_MACRO_UTILS

#ifndef METALERP_FAST
#define METALERP_FAST
#endif

#include "resources/metalerp/metalerp.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#ifdef MLMAX32
    #define ML_PyArray_DType NPY_FLOAT
#elif defined(MLMAX64) 
    #define ML_PyArray_DType NPY_DOUBLE
#elif defined(MLMAX16)
    #define ML_PyArray_DType NPY_FLOAT16
#elif defined(MLMAX_I32)
    #define ML_PyArray_DType NPY_INT32
#elif defined(MLMAX_I64)
    #define ML_PyArray_DType NPY_INT64
#endif

#define PY_METALERP_EXCEPTION_BUFFER_SIZE 1024
//this buffer is currently not thread-safe, multiple lib methods deployed in concurrent threads and firing errors simultaneously can result in issues...
static char exceptionBuffer[PY_METALERP_EXCEPTION_BUFFER_SIZE] = "";

/*CONDITIONS THAT BRANCH TO THIS MUST HAVE SCOPE BRACKETS*/
#define PYMETALERP_RUNTIME_FAILURE(stringMessage, funcName, exception)\
snprintf(exceptionBuffer, PY_METALERP_EXCEPTION_BUFFER_SIZE, "\n\033[4;31mMETALERP EXECUTION FAILURE at: (%s)\033[0m\n****************************\n\033[1;31m%s.\033[0m\nTerminating with NULL return.\n", #funcName, stringMessage);   \
PyErr_SetString(exception, exceptionBuffer);  return NULL


#define PYMETALERP_DEF_COMPUTE_METHOD(methodName, methodInterfaceName, dispatcherName, ssKernelName)    \
static PyObject* methodName(PyObject* self, PyObject* args, PyObject* keywordArgs) {  \
    \
    PyObject *inObj = NULL; \
    BOOL32 cudamode = 0;    \
    \
    static char* keywords[] = {"input_data", "cuda_processing", NULL};  \
    \
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "O|p", keywords, &inObj, &cudamode)) \
        return NULL;    \
    \
    PyArrayObject* inArray; \
    BOOL32 newPyArrayCreated = 0;   \
    \
    if (PyArray_Check(inObj)) /*numpy array*/  \
    {   \
        if (PyArray_TYPE(cast(PyArrayObject*, inObj)) == ML_PyArray_DType)  \
        {   \
                inArray = cast(PyArrayObject*, inObj); \
        }   \
        else    \
        {   \
            \
            PyArray_Descr* nativeDtypeDescr = PyArray_DescrFromType(ML_PyArray_DType);  \
            \
            if (!nativeDtypeDescr)  \
            {   \
                PYMETALERP_RUNTIME_FAILURE("Could not correctly convert the data type of the input numpy array", methodInterfaceName, PyExc_TypeError); \
            }   \
                \
            inArray = cast(PyArrayObject*, PyArray_CastToType(cast(PyArrayObject*, inObj), nativeDtypeDescr, 0));   \
            \
            newPyArrayCreated = 1;  \
        }   \
    }   \
    else if(PyFloat_Check(inObj) || PyLong_Check(inObj)) /*single-scalar*/ \
    {   \
        return PyFloat_FromDouble(ssKernelName(cast(type, PyFloat_AsDouble(inObj))));  \
    }   \
    \
    else  /*generic iterable, auto-casts by default*/  \
    {   \
        inArray = cast(PyArrayObject*, PyArray_FROM_OTF(inObj, ML_PyArray_DType, NPY_ARRAY_IN_ARRAY));  \
        newPyArrayCreated = 1;  \
        \
    }   \
        \
    if (!inArray)  \
    {   \
        PYMETALERP_RUNTIME_FAILURE("Could not parse the input object correctly, you are either passing a non-numeric/non-real-number object (e.g. a string, complex number etc.), an iterable of those (or a key-value iterable, e.g a dictionary), or a jagged iterable (an unevenly dimensional iterable like 2-row list but its first row has 3 columns while the second has only 2) - all of which are not handled by the lib", methodInterfaceName, PyExc_TypeError); \
    }   \
    \
    npy_intp size = PyArray_SIZE(inArray);  \
    if (size <= 0)  \
    {   \
        if(newPyArrayCreated)   \
        { Py_DECREF(inArray); } \
        PYMETALERP_RUNTIME_FAILURE("Input array is empty", methodInterfaceName, PyExc_ValueError);  \
    }   \
    \
    \
    PyArrayObject* outArray = cast (PyArrayObject*, PyArray_SimpleNew(PyArray_NDIM(inArray),    \
                                                                 PyArray_DIMS(inArray), \
                                                                 PyArray_TYPE(inArray)) );  \
    if (!outArray)  \
    {   \
        if(newPyArrayCreated) {Py_DECREF(inArray);} \
        PYMETALERP_RUNTIME_FAILURE("Failed to allocate output array", methodInterfaceName, PyExc_MemoryError);  \
    }   \
    \
    type* inData  = cast(type*, PyArray_DATA(inArray)); \
    type* outData = cast(type*, PyArray_DATA(outArray)); \
    \
    if (size == 1)  \ 
    {   \
        outData[0] = ssKernelName(inData[0]); \
    }   \
    else \
    {   \
        BOOL32 currCUmode = getCUDA_Mode(); \
        BOOL32 cuModeFlagActivatedInScope = 0;  \
        \
        if (cudamode && !currCUmode) {setCUDA_Mode(1); cuModeFlagActivatedInScope = 1;} \
        dispatcherName(inData, outData, size);  \
        if (cuModeFlagActivatedInScope) setCUDA_Mode(0);    \
    }   \
    \
    if(newPyArrayCreated) { Py_DECREF(inArray); }   \
    \
    return cast(PyObject*, outArray);   \
}


#define PYMETALERP_SCALAR_NUMTYPE_CHECK(LP_PyObject_param) ( !PyBool_Check(LP_PyObject_param) && (PyFloat_Check((LP_PyObject_param)) || PyLong_Check((LP_PyObject_param))) )

/*single-argument setter, 4 and 2-argument setters will be written manually*/
#define PYMETALERP_DEF_SETTER(methodName, methodInterfaceName, setterName)  \
static PyObject* methodName(PyObject* self, PyObject* args) \
{   \
    PyObject* parameter = NULL; \
    if (!PyArg_ParseTuple(args, "O", &parameter))   \
        return NULL;    \
    \
    if (PYMETALERP_SCALAR_NUMTYPE_CHECK(parameter)) \
    {   \
        type val = cast(type, PyFloat_AsDouble(parameter));   \
        setterName(val);  \
    }   \
    else \
    {   \
        PYMETALERP_RUNTIME_FAILURE("Expected a scalar number", methodInterfaceName, PyExc_TypeError);   \
    }   \
    \
    Py_RETURN_NONE; \
}

/*multi-arg setters*/

static PyObject* method_setHybridArms(PyObject* self, PyObject* args, PyObject* keywordArgs) 
{
    PyObject *lArm = NULL, *rArm = NULL;

    static char* keywords[] = {"left_arm", "right_arm", NULL};  \
    
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "O!O!", keywords, &PyLong_Type, &lArm, &PyLong_Type, &rArm)) \
        return NULL;    

    enum Functions lArmNum = cast(enum Functions, PyLong_AsLong(lArm)), rArmNum = cast(enum Functions, PyLong_AsLong(rArm));

    if (lArmNum < 0 || lArmNum >= HYBRID_TABLE_SIZE || rArmNum < 0 || rArmNum >= HYBRID_TABLE_SIZE) 
    {
        PYMETALERP_RUNTIME_FAILURE("One or more of the parameters is not a valid, positive, whole number, the function requires both arm inputs to be positive integers and preferably one of the arm constants that the lib provides.\n\
            The setter cannot accept negative numbers or positive ones exceeding the range of the hybrid variant because those numbers are used as indices", setHybridArms, PyExc_TypeError);
    }

    setHybridComboArms(lArmNum, rArmNum);

    Py_RETURN_NONE;
}

static PyObject* method_setHybridArms_LR(PyObject* self, PyObject* args, PyObject* keywordArgs) 
{
    PyObject *lArm = NULL, *rArm = NULL;

    static char* keywords[] = {"left_arm", "right_arm", NULL};  \
    
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "O!O!", keywords, &PyLong_Type, &lArm, &PyLong_Type, &rArm)) \
        return NULL;    

    enum Functions lArmNum = cast(enum Functions, PyLong_AsLong(lArm)), rArmNum = cast(enum Functions, PyLong_AsLong(rArm));

    if (lArmNum < 0 || lArmNum >= HYBRID_LR_TABLE_SIZE || rArmNum < 0 || rArmNum >= HYBRID_LR_TABLE_SIZE) 
    {
        PYMETALERP_RUNTIME_FAILURE("One or more of the parameters is not a valid, positive, whole number, the function requires both arm inputs to be positive integers and preferably one of the arm constants that the lib provides.\n\
            The setter cannot accept negative numbers or positive ones exceeding the range of the hybrid variant because those numbers are used as indices", setHybridArms_LR, PyExc_TypeError);
    }

    setHybridComboArms_LR(lArmNum, rArmNum);

    Py_RETURN_NONE;
}

static PyObject* method_setSigmackParams(PyObject* self, PyObject* args, PyObject* keywordArgs) 
{
    PyObject *min = NULL, *max = NULL, *k = NULL, *v = NULL;

    static char* keywords[] = {"min", "max", "k_parameter", "v_parameter", NULL};  \
    
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "OOOO", keywords, &min, &max, &k, &v)) \
        return NULL;    

    if (!PYMETALERP_SCALAR_NUMTYPE_CHECK(min) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(max) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(k) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(v)) 
    {
        PYMETALERP_RUNTIME_FAILURE("One or more of the parameters is not a valid single-scalar number, the function requires all 4 parameters to be numbers", setSigmackParams, PyExc_TypeError);
    }

    type minNum = cast(type, PyFloat_AsDouble(min)), maxNum = cast(type, PyFloat_AsDouble(max)),
    kNum = cast(type, PyFloat_AsDouble(k)), vNum = cast(type, PyFloat_AsDouble(v));

    setSigmackParams(minNum, maxNum, kNum, vNum);

    Py_RETURN_NONE;
}

static PyObject* method_setNormDistParams(PyObject* self, PyObject* args, PyObject* keywordArgs) 
{
    PyObject *std = NULL, *mean = NULL;

    static char* keywords[] = {"standard_deviation", "mean", NULL};  \
    
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "OO", keywords, &std, &mean)) \
        return NULL;    

    if (!PYMETALERP_SCALAR_NUMTYPE_CHECK(std) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(mean)) 
    {
        PYMETALERP_RUNTIME_FAILURE("One or more of the parameters is not a valid single-scalar number, the function requires standard deviation and mean to be numbers, it is also advised to pass a positive standard deviation", setNormalDistributionParams, PyExc_TypeError);
    }

    type stdNum = cast(type, PyFloat_AsDouble(std)), meanNum = cast(type, PyFloat_AsDouble(mean));

    setNormDistParams(stdNum, meanNum);

    Py_RETURN_NONE;
}

static PyObject* method_setNormDistTunableParams(PyObject* self, PyObject* args, PyObject* keywordArgs) 
{
    PyObject *v = NULL, *minFactor = NULL, *maxFactor = NULL, *kFactor = NULL;

    static char* keywords[] = {"v_parameter", "min_factor", "max_factor", "k_param_factor", NULL};  \
    
    if (!PyArg_ParseTupleAndKeywords(args, keywordArgs, "OOOO", keywords, &v, &minFactor, &maxFactor, &kFactor)) \
        return NULL;    

    if (!PYMETALERP_SCALAR_NUMTYPE_CHECK(v) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(minFactor) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(maxFactor) || !PYMETALERP_SCALAR_NUMTYPE_CHECK(kFactor)) 
    {
        PYMETALERP_RUNTIME_FAILURE("One or more of the parameters is not a valid single-scalar number, the function requires all 4 parameters to be numbers", setNormalDistribution_TunableParams, PyExc_TypeError);
    }

    type vNum = cast(type, PyFloat_AsDouble(v)), minFactorNum = cast(type, PyFloat_AsDouble(minFactor)),
    maxFactorNum = cast(type, PyFloat_AsDouble(maxFactor)), kFactorNum = cast(type, PyFloat_AsDouble(kFactor));

    setNormDistTunableParams(vNum, minFactorNum, maxFactorNum, kFactorNum);

    Py_RETURN_NONE;
}

/*wrappers for state config functions like the cuda mode setter & sign bias setter*/
#define PYMETALERP_DEF_API_CONFIG_METHOD(methodName, methodInterfaceName, configFuncName) 

static PyObject* method_setCudaProcessing(PyObject* self, PyObject* args) 
{
    PyObject* cudamodeObj = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyBool_Type, &cudamodeObj))
        return NULL;

    BOOL32 cudamode = PyObject_IsTrue(cudamodeObj);

    if((cudamode == -1) || PyErr_Occurred())
    {
        PYMETALERP_RUNTIME_FAILURE("The function accepts a boolean value", _setCudaProcessing, PyExc_TypeError);
    }
    
    if(cudamode)
    {
        setCUDA_Mode(1);
    }
    else
    {
        setCUDA_Mode(0);
    }

    Py_RETURN_NONE;
}

static PyObject* method_setSignBias(PyObject* self, PyObject* args) 
{
    PyObject* sgnBiasObj = 0;

    if (!PyArg_ParseTuple(args, "O", &sgnBiasObj))
        return NULL;

    type sgnBias = 1;

    if(!PYMETALERP_SCALAR_NUMTYPE_CHECK(sgnBiasObj) || !( ( (sgnBias = PyFloat_AsDouble(sgnBiasObj)) == cast(type, 1)) ||  (sgnBias == cast(type, -1)) ))
    {
        PYMETALERP_RUNTIME_FAILURE("The number passed to the absolute zero sign bias setter was erroneous, the method accepts either 1 or -1", _setSignBias, PyExc_TypeError);
    }
    setSignBias(sgnBias);
    Py_RETURN_NONE;
}


/*for approximator resetters*/
#define PYMETALERP_DEF_NO_ARG_METHOD(methodName, funcName) \
static PyObject* methodName(PyObject* self, PyObject* Py_UNUSED(args))    \
{   \
    funcName(); \
    Py_RETURN_NONE; \
}



/*compute methods (dispatchers and scalar kernels) interface*/

PYMETALERP_DEF_COMPUTE_METHOD(method_B_A_E, B_A_E, batched_B_A_E, ascendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_B_A_O, B_A_O, batched_B_A_O, ascendingVariant_O)
PYMETALERP_DEF_COMPUTE_METHOD(method_B_D_E, B_D_E, batched_B_D_E, descendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_B_D_O, B_D_O, batched_B_D_O, descendingVariant_O)

PYMETALERP_DEF_COMPUTE_METHOD(method_inv_B_A_E, inv_B_A_E, batched_inv_B_A_E, inv_ascendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_B_A_O, inv_B_A_O, batched_inv_B_A_O, inv_ascendingVariant_O)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_B_D_E, inv_B_D_E, batched_inv_B_D_E, inv_descendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_B_D_O, inv_B_D_O, batched_inv_B_D_O, inv_descendingVariant_O)

PYMETALERP_DEF_COMPUTE_METHOD(method_P_A_E, P_A_E, batched_P_A_E, p_ascendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_P_A_O, P_A_O, batched_P_A_O, p_ascendingVariant_O)
PYMETALERP_DEF_COMPUTE_METHOD(method_P_D_E, P_D_E, batched_P_D_E, p_descendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_P_D_O, P_D_O, batched_P_D_O, p_descendingVariant_O)

PYMETALERP_DEF_COMPUTE_METHOD(method_inv_P_A_E, inv_P_A_E, batched_inv_P_A_E, p_inv_ascendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_P_A_O, inv_P_A_O, batched_inv_P_A_O, p_inv_ascendingVariant_O)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_P_D_E, inv_P_D_E, batched_inv_P_D_E, p_inv_descendingVariant_E)
PYMETALERP_DEF_COMPUTE_METHOD(method_inv_P_D_O, inv_P_D_O, batched_inv_P_D_O, p_inv_descendingVariant_O)

PYMETALERP_DEF_COMPUTE_METHOD(method_Hybrid, Hybrid, batched_Hybrid, hybridVariant)
PYMETALERP_DEF_COMPUTE_METHOD(method_Hybrid_LR, Hybrid_LR, batched_Hybrid_LR, hybridVariant_LR)

PYMETALERP_DEF_COMPUTE_METHOD(method_Sigmack, Sigmack, batched_Sigmack, Sigmack)
PYMETALERP_DEF_COMPUTE_METHOD(method_GaussianDist, NormalDistribution, batched_NormDistApproximator, NormDistApproximator)

//single-param setters
PYMETALERP_DEF_SETTER(method_B_A_E_setMax, B_A_E_setMax, setMaxA_E)
PYMETALERP_DEF_SETTER(method_B_A_E_setMin, B_A_E_setMin, setMinA_E)

PYMETALERP_DEF_SETTER(method_B_A_O_setMax, B_A_O_setMax, setMaxA_O)
PYMETALERP_DEF_SETTER(method_B_A_O_setMin, B_A_O_setMin, setMinA_O)


PYMETALERP_DEF_SETTER(method_B_D_E_setMax, B_D_E_setMax, setMaxD_E)
PYMETALERP_DEF_SETTER(method_B_D_E_setMin, B_D_E_setMin, setMinD_E)

PYMETALERP_DEF_SETTER(method_B_D_O_setMin, B_D_O_setMin, setMinD_O)
PYMETALERP_DEF_SETTER(method_B_D_O_setMax, B_D_O_setMax, setMaxD_O)


PYMETALERP_DEF_SETTER(method_P_A_E_setMax, P_A_E_setMax, p_setMaxA_E)
PYMETALERP_DEF_SETTER(method_P_A_E_setKparam, P_A_E_setKparam, p_setK_A_E)
PYMETALERP_DEF_SETTER(method_P_A_E_setVparam, P_A_E_setVparam, p_setZ_A_E)
PYMETALERP_DEF_SETTER(method_P_A_E_setMin, P_A_E_setMin, p_setMinA_E)


PYMETALERP_DEF_SETTER(method_P_A_O_setMax, P_A_O_setMax, p_setMaxA_O)
PYMETALERP_DEF_SETTER(method_P_A_O_setKparam, P_A_O_setKparam, p_setK_A_O)
PYMETALERP_DEF_SETTER(method_P_A_O_setVparam, P_A_O_setVparam, p_setZ_A_O)
PYMETALERP_DEF_SETTER(method_P_A_O_setMin, P_A_O_setMin, p_setMinA_O)


PYMETALERP_DEF_SETTER(method_P_D_E_setMax, P_D_E_setMax, p_setMaxD_E)
PYMETALERP_DEF_SETTER(method_P_D_E_setKparam, P_D_E_setKparam, p_setK_D_E)
PYMETALERP_DEF_SETTER(method_P_D_E_setVparam, P_D_E_setVparam, p_setZ_D_E)
PYMETALERP_DEF_SETTER(method_P_D_E_setMin, P_D_E_setMin, p_setMinD_E)


PYMETALERP_DEF_SETTER(method_P_D_O_setMax, P_D_O_setMax, p_setMaxD_O)
PYMETALERP_DEF_SETTER(method_P_D_O_setKparam, P_D_O_setKparam, p_setK_D_O)
PYMETALERP_DEF_SETTER(method_P_D_O_setVparam, P_D_O_setVparam, p_setZ_D_O)
PYMETALERP_DEF_SETTER(method_P_D_O_setMin, P_D_O_setMin, p_setMinD_O)

//methods that don't take arguments
PYMETALERP_DEF_NO_ARG_METHOD(method_resetSigmackParams, resetSigmackParams)
PYMETALERP_DEF_NO_ARG_METHOD(method_resetNormalDistributionParams, resetNormDistParams)
PYMETALERP_DEF_NO_ARG_METHOD(method_resetMetalerpGlobalState, metalerp_init)



static PyMethodDef metalerpMethods[] = {
    
    /*lib global state configuration setters*/
    {.ml_name = "setCudaProcessing", .ml_meth=(PyCFunction)method_setCudaProcessing, .ml_flags=METH_VARARGS, .ml_doc="Boolean input, Enable/Disable the lib' CUDA-enabled parallel processing routines if CUDA-capable devices were detected by the library."},
    {.ml_name = "resetMetalerpGlobalState", .ml_meth=(PyCFunction)method_resetMetalerpGlobalState, .ml_flags=METH_NOARGS, .ml_doc="Calls the initialization routine of the library which resets the full global state (variables, indices, cuda mode) to how it were initially when the module was imported."},
    {.ml_name = "setSignBias", .ml_meth=(PyCFunction)method_setSignBias, .ml_flags=METH_VARARGS, .ml_doc="Sets the internal sign function's biasing on absolute zero inputs ~ since it is the simplest way to solve the issue of computing the sign cofactor of x=0."},

    /*global hyperperameter setters*/
    //approximator setters
    {.ml_name = "setSigmackParams", .ml_meth=(PyCFunction)method_setSigmackParams, .ml_flags=METH_VARARGS|METH_KEYWORDS, .ml_doc="Set the Sigmack approximator's parameters, this is for curve fine-tuning - recommended to have the approximators' desmos sheet open while doing this"},
    {.ml_name = "setNormalDistributionParams", .ml_meth=(PyCFunction)method_setNormDistParams, .ml_flags=METH_VARARGS|METH_KEYWORDS, .ml_doc="Set the gaussian approximator's standard deviation and mean in the mentioned order."},
    {.ml_name = "setNormalDistribution_TunableParams", .ml_meth=(PyCFunction)method_setNormDistTunableParams, .ml_flags=METH_VARARGS|METH_KEYWORDS, .ml_doc="Configure the gaussian approximator's backend parameter fusions that influence the curve's shape and height - recommended to have the approximators' desmos sheet open while doing this."},
    
    {.ml_name = "resetSigmackParams", .ml_meth=(PyCFunction)method_resetSigmackParams, .ml_flags=METH_NOARGS, .ml_doc="revert the Sigmack approximator's configurable parameters to the default constants assigned to it at the library's initialization time."},
    {.ml_name = "resetNormalDistributionParams", .ml_meth=(PyCFunction)method_resetNormalDistributionParams, .ml_flags=METH_NOARGS, .ml_doc="revert the Gaussian Distribution approximator's configurable parameters to the default constants assigned to it at the library's initialization time."},
    
    //transform setters
    {.ml_name = "setHybridArms", .ml_meth=(PyCFunction)method_setHybridArms, .ml_flags=METH_VARARGS|METH_KEYWORDS, .ml_doc="Set the hybrid variant's left and right arms, preferably pass to it from the defined metalerp hybrid function table constants."},
    {.ml_name = "setHybridArms_LR", .ml_meth=(PyCFunction)method_setHybridArms_LR, .ml_flags=METH_VARARGS|METH_KEYWORDS, .ml_doc="Set the hybrid left-right variant's left and right arms, preferably pass to it from the defined metalerp hybrid left-right function table constants."},

    {.ml_name = "B_A_E_setMax", .ml_meth=(PyCFunction)method_B_A_E_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding base transform"},
    {.ml_name = "B_A_O_setMax", .ml_meth=(PyCFunction)method_B_A_O_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding base transform"},
    {.ml_name = "B_D_E_setMax", .ml_meth=(PyCFunction)method_B_D_E_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding base transform"},
    {.ml_name = "B_D_O_setMax", .ml_meth=(PyCFunction)method_B_D_O_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding base transform"},

    {.ml_name = "P_A_E_setMax", .ml_meth=(PyCFunction)method_P_A_E_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding parametric transform"},
    {.ml_name = "P_A_O_setMax", .ml_meth=(PyCFunction)method_P_A_O_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding parametric transform"},
    {.ml_name = "P_D_E_setMax", .ml_meth=(PyCFunction)method_P_D_E_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding parametric transform"},
    {.ml_name = "P_D_O_setMax", .ml_meth=(PyCFunction)method_P_D_O_setMax, .ml_flags=METH_VARARGS, .ml_doc="sets the max parameter (b) for the corresponding parametric transform"},

    {.ml_name = "B_A_E_setMin", .ml_meth=(PyCFunction)method_B_A_E_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding base transform"},
    {.ml_name = "B_A_O_setMin", .ml_meth=(PyCFunction)method_B_A_O_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding base transform"},
    {.ml_name = "B_D_E_setMin", .ml_meth=(PyCFunction)method_B_D_E_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding base transform"},
    {.ml_name = "B_D_O_setMin", .ml_meth=(PyCFunction)method_B_D_O_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding base transform"},

    {.ml_name = "P_A_E_setMin", .ml_meth=(PyCFunction)method_P_A_E_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding parametric transform"},
    {.ml_name = "P_A_O_setMin", .ml_meth=(PyCFunction)method_P_A_O_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding parametric transform"},
    {.ml_name = "P_D_E_setMin", .ml_meth=(PyCFunction)method_P_D_E_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding parametric transform"},
    {.ml_name = "P_D_O_setMin", .ml_meth=(PyCFunction)method_P_D_O_setMin, .ml_flags=METH_VARARGS, .ml_doc="sets the min parameter (a) for the corresponding parametric transform"},

    {.ml_name = "P_A_E_setKparam", .ml_meth=(PyCFunction)method_P_A_E_setKparam, .ml_flags=METH_VARARGS, .ml_doc="sets the k tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the K parameter it can be any positive non-zero real number."},
    {.ml_name = "P_A_O_setKparam", .ml_meth=(PyCFunction)method_P_A_O_setKparam, .ml_flags=METH_VARARGS, .ml_doc="sets the k tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the K parameter it can be any positive non-zero real number."},
    {.ml_name = "P_D_E_setKparam", .ml_meth=(PyCFunction)method_P_D_E_setKparam, .ml_flags=METH_VARARGS, .ml_doc="sets the k tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the K parameter it can be any positive non-zero real number."},
    {.ml_name = "P_D_O_setKparam", .ml_meth=(PyCFunction)method_P_D_O_setKparam, .ml_flags=METH_VARARGS, .ml_doc="sets the k tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the K parameter it can be any positive non-zero real number."},

    {.ml_name = "P_A_E_setVparam", .ml_meth=(PyCFunction)method_P_A_E_setVparam, .ml_flags=METH_VARARGS, .ml_doc="sets the v (also interchangeably called z in the desmos sheet page) tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the V parameter it is clamped to any real number equal to or between -1 and 1."},
    {.ml_name = "P_A_O_setVparam", .ml_meth=(PyCFunction)method_P_A_O_setVparam, .ml_flags=METH_VARARGS, .ml_doc="sets the v (also interchangeably called z in the desmos sheet page) tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the V parameter it is clamped to any real number equal to or between -1 and 1."},
    {.ml_name = "P_D_E_setVparam", .ml_meth=(PyCFunction)method_P_D_E_setVparam, .ml_flags=METH_VARARGS, .ml_doc="sets the v (also interchangeably called z in the desmos sheet page) tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the V parameter it is clamped to any real number equal to or between -1 and 1."},
    {.ml_name = "P_D_O_setVparam", .ml_meth=(PyCFunction)method_P_D_O_setVparam, .ml_flags=METH_VARARGS, .ml_doc="sets the v (also interchangeably called z in the desmos sheet page) tuning parameter for the corresponding parametric transform\n\
        This type of setter internally does clamping on the input according to the range that the parameter is allowed to be in, for the V parameter it is clamped to any real number equal to or between -1 and 1."},

    /*transforms*/

    
    {.ml_name = "B_A_E", .ml_meth=(PyCFunction)method_B_A_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="base ascending even variant transform, set its hyperparameters through B_A_E_setMax and B_A_E_setMin methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "B_A_O", .ml_meth=(PyCFunction)method_B_A_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="base ascending odd variant transform, set its hyperparameters through B_A_O_setMax and B_A_O_setMin methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "B_D_E", .ml_meth=(PyCFunction)method_B_D_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="base descending even variant transform, set its hyperparameters through B_D_E_setMax and B_D_E_setMin methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "B_D_O", .ml_meth=(PyCFunction)method_B_D_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="base descending odd variant transform, set its hyperparameters through B_D_O_setMax and B_D_O_setMin methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},

    {.ml_name = "inv_B_A_E", .ml_meth=(PyCFunction)method_inv_B_A_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the base ascending even variant transform, it shares its forward function's parameter space (B_A_E's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_B_A_O", .ml_meth=(PyCFunction)method_inv_B_A_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the base ascending odd variant transform, it shares its forward function's parameter space (B_A_O's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_B_D_E", .ml_meth=(PyCFunction)method_inv_B_D_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the base descending even variant transform, it shares its forward function's parameter space (B_D_E's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_B_D_O", .ml_meth=(PyCFunction)method_inv_B_D_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the base descending odd variant transform, it shares its forward function's parameter space (B_D_O's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},

    {.ml_name = "P_A_E", .ml_meth=(PyCFunction)method_P_A_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="parametric ascending even variant transform, set its hyperparameters through P_A_E_setMax, P_A_E_setMin, P_A_E_setK, and P_A_E_setV methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "P_A_O", .ml_meth=(PyCFunction)method_P_A_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="parametric ascending odd variant transform, set its hyperparameters through P_A_O_setMax and P_A_O_setMin, P_A_O_setK, and P_A_O_setV methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "P_D_E", .ml_meth=(PyCFunction)method_P_D_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="parametric descending even variant transform, set its hyperparameters through P_D_E_setMax and P_D_E_setMin, P_D_E_setK, and P_D_E_setV methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "P_D_O", .ml_meth=(PyCFunction)method_P_D_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="parametric descending odd variant transform, set its hyperparameters through P_D_O_setMax and P_D_O_setMin, P_D_O_setK, and P_D_O_setV methods\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},

    {.ml_name = "inv_P_A_E", .ml_meth=(PyCFunction)method_inv_P_A_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the parametric ascending even variant transform, it shares its forward function's parameter space (P_A_E's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_P_A_O", .ml_meth=(PyCFunction)method_inv_P_A_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the parametric ascending odd variant transform, it shares its forward function's parameter space (P_A_O's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_P_D_E", .ml_meth=(PyCFunction)method_inv_P_D_E, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the parametric descending even variant transform, it shares its forward function's parameter space (P_D_E's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "inv_P_D_O", .ml_meth=(PyCFunction)method_inv_P_D_O, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="inverse of the parametric descending odd variant transform, it shares its forward function's parameter space (P_D_O's setters)\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},

    {.ml_name = "Hybrid", .ml_meth=(PyCFunction)method_Hybrid, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="The hybrid variant transform: enables combining the behavior of different transforms (2 at a time) by choosing left arm (which transform it will behave like for x<0) and right arm (which transform it will behave like for x>=0) parameters, it references the other transforms for execution so you only need to globally set the parameters of the specific transforms chosen\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "Hybrid_LR", .ml_meth=(PyCFunction)method_Hybrid_LR, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="The hybrid variant Left-Right transform: behaves like the hybrid variant with additional support for the full transformation space in any of the arms - meaning that left arms can be used for right-arm behavior and vice-versa too; this function can easily produce weird and non-deterministic results or invalid numbers if you do not know what you are doing, so it is advised that you keep the desmos sheets opened while using it to analyze the behavior of the transforms you'll be combining\nAdditionally, it does the same as metalerp.Hybrid() function (strongly advise looking at its documentation) for transform referencing\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},


    /*approximators*/
    {.ml_name = "Sigmack", .ml_meth=(PyCFunction)method_Sigmack, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="The Sigmack ( Sig(moid) + (Car)mack, named after John D. Carmack).\nA fast, accurate, and configurable approximation of the Sigmoid (Logistic/Normal Distribution's CDF) Function\nIf you need to revert to the exact default parameters of this approximator, you can easily reset with the resetSigmackParams() method, just be aware that it will also reset the standard deviation and mean parameters of the function\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},
    {.ml_name = "NormalDistribution", .ml_meth=(PyCFunction)method_GaussianDist, .ml_flags=METH_VARARGS | METH_KEYWORDS, .ml_doc="A fast, accurate, and configurable approximation function of the Gaussian Distribution's PDF\nIf you need to revert to the exact default parameters of this approximator, you can easily reset with the resetNormDistParams() method, just be aware that it will also reset the standard deviation and mean parameters of the function\n\
         has device processing enabled via calling set_cuda_processing which sets the mode globally for the metalerp module or by optionally passing True/False for the second argument"},


    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef metalerpModule = {
    PyModuleDef_HEAD_INIT, "metalerp",
    "metalerp", -1, metalerpMethods
};


uint32_t metalerp_globalArr_index = 0;
#define METALERP_ADD_GLOBAL(call) (moduleGlobalVariables[metalerp_globalArr_index++] = (call));
#define PYMETALERP_GLOBAL_COUNT 40
PyMODINIT_FUNC PyInit_metalerp(void) 
{
    
    printf("Thank you for using metalerp, for easy analytical visualizations of the transformations provided, please visit:\n\t%s  (for the base and parametric transforms and their inverses)\nand     %s  (for the approximators)\n\n",
          "https://www.desmos.com/calculator/cf0389db8e", "https://www.desmos.com/calculator/neji45kf1n");
    
    METALERP_INIT
    
    import_array(); 

    //lib global variables 

    PyObject* nativeDtype = cast(PyObject*, PyArray_DescrFromType(ML_PyArray_DType));

    
    if(!nativeDtype) return NULL;

    PyObject* module = PyModule_Create(&metalerpModule);

    int moduleGlobalVariables[PYMETALERP_GLOBAL_COUNT] = {-1};


    METALERP_ADD_GLOBAL(PyModule_AddObject(module, "metalerp_dtype", nativeDtype))
    

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_B_A_E", B_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_B_A_O", B_ASC_ODD))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_B_D_E", B_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_B_D_O", B_DESC_ODD))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_P_A_E", P_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_P_A_O", P_ASC_ODD))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_P_D_E", P_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_P_D_O", P_DESC_ODD))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_B_A_E", B_INV_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_B_A_O", B_INV_ASC_ODD))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_B_D_E", B_INV_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_B_D_O", B_INV_DESC_ODD))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_P_A_E", P_INV_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_P_A_O", P_INV_ASC_ODD))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_P_D_E", P_INV_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_inv_P_D_O", P_INV_DESC_ODD))


    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_A_E", LR_B_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_A_O_Left", LR_B_ASC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_A_O_Right", LR_B_ASC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_D_E", LR_B_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_D_O_Left", LR_B_DESC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_B_D_O_Right", LR_B_DESC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_A_E", LR_P_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_A_O_Left", LR_P_ASC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_A_O_Right", LR_P_ASC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_D_E", LR_P_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_D_O_Left", LR_P_DESC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_P_D_O_Right", LR_P_DESC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_A_E", LR_B_INV_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_A_O_Left", LR_B_INV_ASC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_A_O_Right", LR_B_INV_ASC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_D_E", LR_B_INV_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_D_O_Left", LR_B_INV_DESC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_B_D_O_Right", LR_B_INV_DESC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_A_E", LR_P_INV_ASC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_A_O_Left", LR_P_INV_ASC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_A_O_Right", LR_P_INV_ASC_ODD_R))

    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_D_E", LR_P_INV_DESC_EVEN))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_D_O_Left", LR_P_INV_DESC_ODD_L))
    METALERP_ADD_GLOBAL(PyModule_AddIntConstant(module, "arm_LR_inv_P_D_O_Right", LR_P_INV_DESC_ODD_R))

    BOOL32 negativeDetected = 0;
    
    for(int32_t i=1; i<(sizeof(moduleGlobalVariables) / sizeof(moduleGlobalVariables[0])); ++i)
    {
        if(moduleGlobalVariables[i] < 0)
            negativeDetected = 1;
    }

    if(moduleGlobalVariables[0] < 0)
    {
        Py_DECREF(nativeDtype);
        Py_DECREF(module);
        return NULL;
    }
    
    if(negativeDetected)
    {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}