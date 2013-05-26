#ifndef _RAIJIN_COMPLEX_H
#define _RAIJIN_COMPLEX_H
#include "raijin.hpp"

namespace RaijinCL{
typedef RaijinGemm<cl_double2> RaijinZgemm;
typedef RaijinGemm<cl_float2> RaijinCgemm;
void raijinTuneZgemm(cl_device_id dvc);
void raijinTuneCgemm(cl_device_id dvc);

}

#endif
