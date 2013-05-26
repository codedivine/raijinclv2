#include "raijin_complex.hpp"
#include <fstream>
using namespace RaijinCL;
using namespace std;

static string buildComplexGenKernel(bool isDouble){
    stringstream ss;
    if(isDouble) ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    string dtype = (isDouble) ? "double" : "float";
    string kname = (isDouble) ? "zgemmGen" : "cgemmGen";
    ss<<"__kernel void "<<kname<<"(int K, "<<dtype<<"2 alpha, ";
    ss<<"const "<<dtype<<"2 __global *A, unsigned int lda, int offsetA, ";
    ss<<"const "<<dtype<<"2 __global *B, unsigned int ldb, int offsetB, ";
    ss<<dtype<<"2 beta,";
    ss<<" __global "<<dtype<<"2 *C, unsigned int ldc, int offsetC){\n";
    ss<<" int i = get_global_id(1);\n";
    ss<<" int j = get_global_id(0);\n";
    ss<<" int k;\n";
    ss<<dtype<<"2 sum;";
    ss<<" sum.x = 0.0;";
    ss<<" sum.y = 0.0;";
    ss<<" for(k=0;k<K;k++){";
    ss<<"   "<<dtype<<"2 atemp = A[i*lda+k+offsetA]; ";
    ss<<"   "<<dtype<<"2 btemp = B[k*ldb + j+offsetB]";
    ss<<"	sum.x += (atemp.x*btemp.x - atemp.y*btemp.y);";
    ss<<"   sum.y += (atemp.x*btemp.y + atemp.y*btemp.x);";
    ss<<" }";
    ss<<" "<<dtype<<"2 ctemp = C[i*ldc + j+offsetC];";
    ss<<" "<<dtype<<"2 temp;";
    ss<<" temp.x = alpha.x*sum.x - alpha.y*sum.y + beta.x*ctemp.x - beta.y*ctemp.y;";
    ss<<" temp.y = alpha.x*sum.y + alpha.y*sum.x + beta.x*ctemp.y + beta.y*ctemp.x;";
    ss<<" C[i*ldc + j+offsetC] = temp;";
    ss<<"}";
    return ss.str();
}

template <typename T>
T getOneComplex(){
    T val;
    val.s[0] = 1;
    val.s[1] = 0;
    return val;
}

template <>
std::string RaijinCL::getGenGemmKernel<cl_float2>(){
    return buildComplexGenKernel(false);
}

template <>
std::string RaijinCL::getGenGemmKernel<cl_double2>(){
    return buildComplexGenKernel(true);
}

template <>
std::string RaijinCL::getGemmname<cl_float2>(){
    return "cgemm";
}

template <>
std::string RaijinCL::getGemmname<cl_double2>(){
    return "zgemm";
}

template <>
cl_double2 RaijinCL::raijinGetOne(){
    return getOneComplex<cl_double2>();
}

template <>
cl_float2 RaijinCL::raijinGetOne(){
    return getOneComplex<cl_float2>();
}
