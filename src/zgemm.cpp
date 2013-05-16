#include "raijin_complex.hpp"
#include <fstream>
using namespace RaijinCL;
using namespace std;

static const string zgemmGenKernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "__kernel void dgemmGen(int K, double2 alpha, "
        "const double2 __global *A, unsigned int lda, int offsetA, "
        "const double2 __global *B, unsigned int ldb, int offsetB, "
        "double2 beta,"
        " __global double2 *C, unsigned int ldc, int offsetC){\n"
        " int i = get_global_id(1);\n"
        " int j = get_global_id(0);\n"
        " int k;\n"
        " double2 sum;"
        " sum.x = 0.0;"
        " sum.y = 0.0;"
        " for(k=0;k<K;k++){"
        "   double2 atemp = A[i*lda+k+offsetA]; "
        "   double2 btemp = B[k*ldb + j+offsetB]"
        "	sum.x += (atemp.x*btemp.x - atemp.y*btemp.y);"
        "   sum.y += (atemp.x*btemp.y + atemp.y*btemp.x);"
        " }"
        " double2 ctemp = C[i*ldc + j+offsetC];"
        " double2 temp;"
        " temp.x = alpha.x*sum.x - alpha.y*sum.y + beta.x*ctemp.x - beta.y*ctemp.y;"
        " temp.y = alpha.x*sum.y + alpha.y*sum.x + beta.x*ctemp.y + beta.y*ctemp.x;"
        " C[i*ldc + j+offsetC] = temp;"
        "}";


cl_event RaijinZgemm::applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_double2 alpha, cl_mem A, cl_uint lda,
        cl_mem B, cl_uint ldb,
        cl_double2 beta,
        cl_mem C, cl_uint ldc, RaijinParams params){
    return raijinApplyOpt<cl_double2>(optcompiled,optkernel,ctx,dvc,
                          order,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,params,this->transObj,this->copyObj,this->scaleObj);

}

RaijinZgemm *RaijinZgemm::getInstance(cl_context context,cl_device_id device){
    string fname = raijinGetProfileFileName(device,"zgemm");
    RaijinGemmOptKernel opts;
    ifstream ifile(fname.c_str());
    if(!ifile.good() || !ifile.is_open()){
        cout<<"Could not open ZGEMM profile "<<fname<<endl;
    }
    ifile>>opts;
    RaijinZgemm *zgemm = new RaijinZgemm;
    zgemm->optkernel = opts;
    zgemm->ctx = context;
    zgemm->dvc = device;
    cl_int errcode;
    const size_t len = zgemm->optkernel.kernel.length();
    const char *prgstr = zgemm->optkernel.kernel.c_str();
    //cout<<"Kernel:"<<sgemm->optkernel.kernel<<endl;

    //TODO: check that there were no errors
    zgemm->optprg = clCreateProgramWithSource(zgemm->ctx, 1, &prgstr, &len, &errcode);
    //cout<<"Create program from source "<<errcode<<endl;
    cl_int bldcode = clBuildProgram(zgemm->optprg, 1, &(zgemm->dvc), "", NULL, NULL);
    //cout<<"Build code "<<bldcode<<endl;
    zgemm->optcompiled = clCreateKernel(zgemm->optprg, zgemm->optkernel.kname.c_str(), &errcode);
    zgemm->transObj = new RaijinTranspose(device,context);
    zgemm->copyObj = new RaijinCopy(context,device);
    zgemm->scaleObj = new RaijinScale(context,device);
    //printProgramBinary(zgemm->optprg);
    //TODO: build the generic kernel
    return zgemm;
}

cl_event RaijinZgemm::apply(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_double2 alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
        cl_mem B, cl_uint ldb, cl_uint offsetB,
        cl_double2 beta,
        cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params){
    const int elemsize = sizeof(cl_double2);

    if(order==RaijinColMajor){
        //untested
        return apply(RaijinRowMajor,transA,transB,N,M,K,alpha,B,ldb,offsetB,A,lda,offsetA,beta,C,ldc,offsetC,params);
    }
    //cout<<"Entering apply"<<endl;
    cl_mem optA,optB,optC;
    int optM,optN,optK;
    optM = M - M%(optkernel.htile*optkernel.lsizex);
    optK = K - K%(optkernel.ktile);
    optN = N - N%(optkernel.wtile*optkernel.lsizey);
    //cout<<optM<<" "<<optN<<" "<<optK<<" "<<optkernel.ktile<<endl;
    cl_int errcode;
    RaijinParams temp;
    temp.queue = params.queue;
    temp.waitEvents = new cl_event[params.num_events+4];
    temp.num_events = params.num_events;
    for(int i=0;i<params.num_events;i++) temp.waitEvents[i] = params.waitEvents[i];
    if(optM>0 && optN>0 && optK>0){
        if(offsetA>0){
            cl_buffer_region regA;
            regA.origin = offsetA;
            regA.size = optM*lda*elemsize;
            optA = clCreateSubBuffer(A,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regA,&errcode);
        }else{
            optA = A;
        }
        if(offsetB>0){
            cl_buffer_region regB;
            regB.origin = offsetB;
            regB.size = optK*ldb*elemsize;
            optB = clCreateSubBuffer(B,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regB,&errcode);
        }else{
            optB = B;
        }
        if(offsetC>0){
            cl_buffer_region regC;
            regC.origin = offsetC;
            regC.size = optM*ldc*elemsize;
            optC = clCreateSubBuffer(C,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regC,&errcode);
        }else{
            optC = C;
        }

        cl_event evt = applyOpt(order,transA,transB,optM,optN,optK,alpha,optA,lda,optB,ldb,beta,optC,ldc,temp);
        if(offsetA>0) clReleaseMemObject(optA);
        if(offsetB>0) clReleaseMemObject(optB);
        if(offsetC>0) clReleaseMemObject(optC);
        temp.waitEvents[temp.num_events] = evt;
        temp.num_events++;
    }

    cl_event lastEvt = temp.waitEvents[temp.num_events-1];
    delete[] temp.waitEvents;
    return lastEvt;

}

RaijinZgemm::~RaijinZgemm(){
    clReleaseKernel(optcompiled);
    clReleaseProgram(optprg);
    clReleaseContext(ctx);
    delete scaleObj;
    delete transObj;
    delete copyObj;
}

RaijinCgemm::~RaijinCgemm(){
    clReleaseKernel(optcompiled);
    clReleaseProgram(optprg);
    clReleaseContext(ctx);
    delete scaleObj;
    delete transObj;
    delete copyObj;
}

RaijinCgemm *RaijinCgemm::getInstance(cl_context context,cl_device_id device){
    string fname = raijinGetProfileFileName(device,"cgemm");
    RaijinGemmOptKernel opts;
    ifstream ifile(fname.c_str());
    if(!ifile.good() || !ifile.is_open()){
        cout<<"Could not open CGEMM profile "<<fname<<endl;
    }
    ifile>>opts;
    RaijinCgemm *cgemm = new RaijinCgemm;
    cgemm->optkernel = opts;
    cgemm->ctx = context;
    cgemm->dvc = device;
    cl_int errcode;
    const size_t len = cgemm->optkernel.kernel.length();
    const char *prgstr = cgemm->optkernel.kernel.c_str();
    //cout<<"Kernel:"<<sgemm->optkernel.kernel<<endl;

    //TODO: check that there were no errors
    cgemm->optprg = clCreateProgramWithSource(cgemm->ctx, 1, &prgstr, &len, &errcode);
    //cout<<"Create program from source "<<errcode<<endl;
    cl_int bldcode = clBuildProgram(cgemm->optprg, 1, &(cgemm->dvc), "", NULL, NULL);
    //cout<<"Build code "<<bldcode<<endl;
    cgemm->optcompiled = clCreateKernel(cgemm->optprg, cgemm->optkernel.kname.c_str(), &errcode);
    cgemm->transObj = new RaijinTranspose(device,context);
    cgemm->copyObj = new RaijinCopy(context,device);
    cgemm->scaleObj = new RaijinScale(context,device);
    //printProgramBinary(cgemm->optprg);
    //TODO: build the generic kernel
    return cgemm;
}

cl_event RaijinCgemm::apply(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_float2 alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
        cl_mem B, cl_uint ldb, cl_uint offsetB,
        cl_float2 beta,
        cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params){
    const int elemsize = sizeof(cl_double2);

    if(order==RaijinColMajor){
        //untested
        return apply(RaijinRowMajor,transA,transB,N,M,K,alpha,B,ldb,offsetB,A,lda,offsetA,beta,C,ldc,offsetC,params);
    }
    //cout<<"Entering apply"<<endl;
    cl_mem optA,optB,optC;
    int optM,optN,optK;
    optM = M - M%(optkernel.htile*optkernel.lsizex);
    optK = K - K%(optkernel.ktile);
    optN = N - N%(optkernel.wtile*optkernel.lsizey);
    //cout<<optM<<" "<<optN<<" "<<optK<<" "<<optkernel.ktile<<endl;
    cl_int errcode;
    RaijinParams temp;
    temp.queue = params.queue;
    temp.waitEvents = new cl_event[params.num_events+4];
    temp.num_events = params.num_events;
    for(int i=0;i<params.num_events;i++) temp.waitEvents[i] = params.waitEvents[i];
    if(optM>0 && optN>0 && optK>0){
        if(offsetA>0){
            cl_buffer_region regA;
            regA.origin = offsetA;
            regA.size = optM*lda*elemsize;
            optA = clCreateSubBuffer(A,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regA,&errcode);
        }else{
            optA = A;
        }
        if(offsetB>0){
            cl_buffer_region regB;
            regB.origin = offsetB;
            regB.size = optK*ldb*elemsize;
            optB = clCreateSubBuffer(B,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regB,&errcode);
        }else{
            optB = B;
        }
        if(offsetC>0){
            cl_buffer_region regC;
            regC.origin = offsetC;
            regC.size = optM*ldc*elemsize;
            optC = clCreateSubBuffer(C,0,CL_BUFFER_CREATE_TYPE_REGION,(const void*)&regC,&errcode);
        }else{
            optC = C;
        }

        cl_event evt = applyOpt(order,transA,transB,optM,optN,optK,alpha,optA,lda,optB,ldb,beta,optC,ldc,temp);
        if(offsetA>0) clReleaseMemObject(optA);
        if(offsetB>0) clReleaseMemObject(optB);
        if(offsetC>0) clReleaseMemObject(optC);
        temp.waitEvents[temp.num_events] = evt;
        temp.num_events++;
    }

    cl_event lastEvt = temp.waitEvents[temp.num_events-1];
    delete[] temp.waitEvents;
    return lastEvt;

}

cl_event RaijinCgemm::applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_float2 alpha, cl_mem A, cl_uint lda,
        cl_mem B, cl_uint ldb,
        cl_float2 beta,
        cl_mem C, cl_uint ldc, RaijinParams params){
    return raijinApplyOpt<cl_float2>(optcompiled,optkernel,ctx,dvc,
                          order,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,params,this->transObj,this->copyObj,this->scaleObj);

}

template <typename T>
T getOneComplex(){
    T val;
    val.s[0] = 1;
    val.s[1] = 0;
    return val;
}

template <>
cl_double2 RaijinCL::raijinGetOne(){
    return getOneComplex<cl_double2>();
}

template <>
cl_float2 RaijinCL::raijinGetOne(){
    return getOneComplex<cl_float2>();
}
