#ifndef _RAIJIN_COMPLEX_H
#define _RAIJIN_COMPLEX_H
#include "raijin.hpp"

namespace RaijinCL{
class RaijinZgemm{
public:
    static RaijinZgemm *getInstance(cl_context ctx,cl_device_id dvc);
    cl_event apply(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            cl_double2 alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
            cl_mem B, cl_uint ldb, cl_uint offsetB,
            cl_double2 beta,
            cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params);
    ~RaijinZgemm();
private:
    RaijinZgemm(){}
    RaijinZgemm(const RaijinZgemm& );
    cl_event applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            cl_double2 alpha, cl_mem A, cl_uint lda,
            cl_mem B, cl_uint ldb,
            cl_double2 beta,
            cl_mem C, cl_uint ldc, RaijinParams params);
    RaijinGemmOptKernel optkernel;
    cl_context ctx;
    cl_device_id dvc;
    cl_kernel optcompiled;
    cl_program optprg;
    RaijinTranspose *transObj;
    RaijinCopy *copyObj;
    RaijinScale *scaleObj;
};

class RaijinCgemm{
public:
    static RaijinCgemm *getInstance(cl_context ctx,cl_device_id dvc);
    cl_event apply(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            cl_float2 alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
            cl_mem B, cl_uint ldb, cl_uint offsetB,
            cl_float2 beta,
            cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params);
    ~RaijinCgemm();
private:
    RaijinCgemm(){}
    RaijinCgemm(const RaijinZgemm& );
    cl_event applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            cl_float2 alpha, cl_mem A, cl_uint lda,
            cl_mem B, cl_uint ldb,
            cl_float2 beta,
            cl_mem C, cl_uint ldc, RaijinParams params);
    RaijinGemmOptKernel optkernel;
    cl_context ctx;
    cl_device_id dvc;
    cl_kernel optcompiled;
    cl_program optprg;
    RaijinTranspose *transObj;
    RaijinCopy *copyObj;
    RaijinScale *scaleObj;
};


void raijinTuneZgemm(cl_device_id dvc);
void raijinTuneCgemm(cl_device_id dvc);


}

#endif
