/**Copyright 2012, Rahul Garg and McGill University
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef RAIJINCL_H
#define RAIJINCL_H

#define RAIJIN_EXPERIMENTAL
#ifdef RAIJIN_EXPERIMENTAL

template <typename T> void BlasFun(void* params);

extern "C"{
#include <acml.h>
}

template <typename T>
struct BlasParams{
	//CBLAS_ORDER order;
	bool transA;
	bool transB;
	int M;
	int N;
	int K;
	T alpha;
	T *A;
	int lda;
	T *B;
	int ldb;
	T beta;
	T *C;
	int ldc;
};

template <>
void BlasFun<float>(void* params);

template<>
void BlasFun<double>(void* params);
#endif



#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
extern "C"{
#include <CL/cl.h>
}
#include <string>
#include <sstream>
#include <iostream>
#include <istream>
#include <ostream>
#include <fstream>
#include <vector>
#include "rtimer.hpp"

namespace RaijinCL{

enum RAIJIN_SUCCESS {RaijinSuccess, RaijinError};
enum RAIJIN_TRANSPOSE {RaijinNoTrans, RaijinTrans};
enum RAIJIN_ORDER {RaijinRowMajor, RaijinColMajor};

std::string raijinGetProfileFileName(cl_device_id dvc,std::string prefix="");

/*! This class is used for storing information about the kernels found by the tuner */
struct RaijinGemmOptKernel{
	std::string kernel;
	std::string kname;
	std::string dvcname;
	int lsizex;
	int lsizey;
	int htile;
	int wtile;
	int ktile;
    int simdwidth;
    bool transA;
    bool transB;
	bool imageA;
	bool imageB;
};

std::istream& operator>>(std::istream &stream,RaijinGemmOptKernel& krnl);
std::ostream& operator<<(std::ostream &stream,RaijinGemmOptKernel krnl);

class RaijinCopy{
    cl_context ctx;
    cl_device_id dvc;
    cl_kernel sbuf,dbuf,cbuf,zbuf;
    cl_kernel simg[3],dimg[2],cimg[2],zimg;
public:
    RaijinCopy(cl_context context,cl_device_id device);
    ~RaijinCopy();
    cl_event scopyToBuf(cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event scopyToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event dcopyToBuf(cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event dcopyToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event ccopyToBuf(cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event ccopyToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event zcopyToBuf(cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event zcopyToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
};

template <typename T>
cl_event raijinCopyToBuf(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToBuf<cl_float>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToBuf<cl_float2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToBuf<cl_double>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToBuf<cl_double2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda);

template <typename T>
cl_event raijinCopyToImg(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToImg<cl_float>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToImg<cl_double>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToImg<cl_float2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event raijinCopyToImg<cl_double2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

class RaijinScale{
    cl_context ctx;
    cl_device_id dvc;
    cl_kernel skrnl,dkrnl,ckrnl,zkrnl;
public:
    RaijinScale(cl_context ctx,cl_device_id dvc);
    cl_event sscale(cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_float beta);
    cl_event dscale(cl_command_queue q,cl_mem C,int M,int N,int ldc,cl_double beta);
    cl_event cscale(cl_command_queue q,cl_mem C,int M,int N,int ldc,cl_float2 beta);
    cl_event zscale(cl_command_queue q,cl_mem C,int M,int N,int ldc,cl_double2 beta);
};

template <typename T>
cl_event raijinScale(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,T beta);

template <>
cl_event raijinScale<cl_float>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_float beta);

template <>
cl_event raijinScale<cl_float2>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_float2 beta);

template <>
cl_event raijinScale<cl_double>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_double beta);

template <>
cl_event raijinScale<cl_double2>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_double2 beta);



struct RaijinTransOpt{
    int lx;
    int ly;
    std::string kernel;
};

std::istream& operator>>(std::istream &stream,RaijinTransOpt& krnl);
std::ostream& operator<<(std::ostream &stream,RaijinTransOpt krnl);

class RaijinTranspose{
public:
    RaijinTranspose(cl_device_id device,cl_context context);
    cl_event stransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event stransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event dtransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event dtransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event ctransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event ctransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

    cl_event ztransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
    cl_event ztransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
    static void tuneStrans(cl_device_id dvc);
    static void tuneDtrans(cl_device_id dvc);
    static void tuneCtrans(cl_device_id dvc);
    static void tuneZtrans(cl_device_id dvc);
    ~RaijinTranspose();
private:
    RaijinTransOpt sparams[2][4];
    cl_kernel skernels[2][4];
    cl_kernel sgkernels[2][4];
    bool sinit[2][4];

    RaijinTransOpt dparams[2][4];
    cl_kernel dkernels[2][4];
    cl_kernel dgkernels[2][4];
    bool dinit[2][4];

    RaijinTransOpt cparams[2][4];
    cl_kernel ckernels[2][4];
    cl_kernel cgkernels[2][4];
    bool cinit[2][4];

    RaijinTransOpt zparams[2][4];
    cl_kernel zkernels[2][4];
    cl_kernel zgkernels[2][4];
    bool zinit[2][4];

    cl_device_id dvc;
    cl_context ctx;
    static bool createRealTrans(int simdw,int lx,int ly,bool useLocalMem,bool toImg,bool isDouble,std::string& res,bool insertAttrib=true);
    static bool createComplexTrans(int simdw,int lx,int ly,bool useLocalMem,bool toImg,bool isDouble,std::string& res,bool insertAttrib=true);

};

template <typename T>
cl_event transToBuf(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToBuf<cl_float>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToBuf<cl_double>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToBuf<cl_float2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToBuf<cl_double2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);


template <typename T>
cl_event transToImg(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToImg<cl_float>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToImg<cl_double>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToImg<cl_float2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

template <>
cl_event transToImg<cl_double2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);

void raijinTuneSgemm(cl_device_id dvc);
void raijinTuneDgemm(cl_device_id dvc);


extern void writeResults(std::string fname,cl_device_id dvc,std::string& prgmsource,std::string& prgmname,int tsizes[],int lsizes[],bool transA,bool transB);

size_t findMaxDivisor(size_t val,size_t max);
void releaseMemObject(cl_event evt,cl_int status,void *data);

void printProgramBinary(cl_program prg);

class RaijinGemmPlan{
public:
struct TileSpace{
	std::vector<int> ivals;
	std::vector<int> jvals;
	std::vector<int> kvals;
};



struct ExecBufs{
    ~ExecBufs();
	std::vector<cl_mem> memobj;

    //whether this was allocated, or not
	std::vector<bool> isAllocated;

    //if not allocated, then this is either A, or copy of a previuos buffer
    //if is allocated, then doesn't matter
    std::vector<bool> isPrevBuf;

    //whether this is a copy, or a transpose. only matters if isAllocated is true
	std::vector<bool> isCopy;

	size_t allocBytes;
};

ExecBufs abufs,bbufs;
TileSpace tiles;
cl_uint M,N,K,lda,ldb;
cl_context ctx;
cl_device_id dvc;
bool transA,transB;
RaijinGemmOptKernel optparams;
int msize;
static void deleteGemmPlan(cl_event evt,cl_int status,void *vplan);
};

struct RaijinCleaner{
	std::vector<cl_mem> bufs;
	RaijinGemmPlan *plan;
	RaijinCleaner():plan(NULL){}
	~RaijinCleaner();
};


template <typename T>
bool raijinIsDouble(){return false;}

template<>
bool raijinIsDouble<cl_double2>();

template<>
bool raijinIsDouble<cl_double>();

template <typename T>
void raijinPrivateAllocStuff(RaijinGemmPlan::ExecBufs& execbuf,
                             cl_context ctx,
                             bool transA,
                             bool opttransA,
                             int simd,
                             cl_uint lda,
                             int imax,
                             int kmax,
                             const std::vector<int>& ivals,
                             const std::vector<int>& kvals,
                             bool useImageA,
                             int msize){
    using namespace std;
    const bool isDouble = raijinIsDouble<T>();
    vector<cl_mem> abufsMem(imax*kmax);
    vector<bool> abufsAlloc(imax*kmax);
    vector<bool> abufsCopy(imax*kmax);
    vector<bool> abufsPrevBuf(imax*kmax);
    execbuf.allocBytes = 0;

    for(int i=0;i<imax;i++){
        for(int k=0;k<kmax;k++){
            size_t index = i*kmax+k;

            //for now, we do not reuse buffers
            abufsPrevBuf[index] = false;

            const cl_uint Mnew = ivals[i+1]-ivals[i];
            const cl_uint Knew = kvals[k+1]-kvals[k];
            cl_mem Anew;
            bool isAlloc = true;
            if(!useImageA && opttransA==transA && lda<2*msize && imax==1 && kmax==1){
                isAlloc = false;
            }

            const bool isCopy = (opttransA == transA);
            abufsAlloc[index] = isAlloc;
            abufsCopy[index] = isCopy;

            if(isAlloc){
                if(useImageA){
                    int ncomps = sizeof(T)*simd/sizeof(cl_float);
                    cl_image_format format;
                    switch(ncomps){
                    case 1:
                        format.image_channel_order = CL_R;
                        break;
                    case 2:
                        format.image_channel_order = CL_RG;
                        break;
                    case 4:
                        format.image_channel_order = CL_RGBA;
                        break;
                    }
                    if(isDouble) format.image_channel_data_type = CL_SIGNED_INT32;
                    else format.image_channel_data_type = CL_FLOAT;
                    size_t imgheight,imgwidth;
                    if(isCopy){
                        imgheight = Mnew;
                        imgwidth = Knew/simd;
                    }else{
                        imgheight = Knew;
                        imgwidth = Mnew/simd;
                    }
                    cl_mem img = clCreateImage2D(ctx,CL_MEM_READ_WRITE,&format,imgwidth,imgheight,0,NULL,NULL);
                    abufsMem[index] = img;
                }else{
                    size_t bufsize = sizeof(T)*Mnew*Knew;
                    //cout<<"raijinPrivateAllocStuff: Created buffer of size "<<bufsize<<endl;
                    cl_mem buf = clCreateBuffer(ctx,CL_MEM_READ_WRITE,bufsize,NULL,NULL);
                    abufsMem[index] = buf;
                }
                execbuf.allocBytes += sizeof(T)*Mnew*Knew;
            }else{
                abufsCopy[index] = false;
            }
        }
    }
    execbuf.memobj = abufsMem;
    execbuf.isAllocated = abufsAlloc;
    execbuf.isCopy = abufsCopy;
    execbuf.isPrevBuf = abufsPrevBuf;

}

template <typename T>
RaijinGemmPlan *raijinGetGemmPlan(RaijinGemmOptKernel optparams,
                                  cl_context ctx,
                                  cl_device_id dvc,
                                  enum RAIJIN_ORDER order,
                                  bool transA,
                                  bool transB,
                                  cl_uint M,
                                  cl_uint N,
                                  cl_uint K,
                                  cl_uint lda,
                                  cl_uint ldb,int msize){
    using namespace std;
    const bool isDouble = raijinIsDouble<T>();
    RaijinGemmPlan *plan = new RaijinGemmPlan;
    plan->transA = transA;
    plan->transB = transB;
    plan->M = M;
    plan->N = N;
    plan->K = K;
    plan->lda = lda;
    plan->ldb = ldb;
    plan->optparams = optparams;
    plan->ctx = ctx;
    plan->dvc = dvc;
    plan->msize = msize;

    //cout<<"Inside applyOpt"<<endl;
    const size_t elemsize = sizeof(T);
    cl_device_type dvctype;
    clGetDeviceInfo(dvc,CL_DEVICE_TYPE,sizeof(dvctype),&dvctype,NULL);
    int imax = M/msize;
    int jmax = N/msize;
    int kmax = K/msize;
    bool useImageA = optparams.imageA;
    bool useImageB = optparams.imageB;

    if(imax==0) imax = 1;
    if(jmax==0) jmax = 1;
    if(kmax==0) kmax = 1;

    vector<int> ivals(imax+1);
    vector<int> jvals(jmax+1);
    vector<int> kvals(kmax+1);
    for(int i=0;i<imax;i++) ivals[i] = i*msize;
    ivals[imax] = M;

    for(int j=0;j<jmax;j++) jvals[j] = j*msize;
    jvals[jmax] = N;

    for(int k=0;k<kmax;k++) kvals[k] = k*msize;
    kvals[kmax] = K;
    plan->tiles.ivals = ivals;
    plan->tiles.jvals = jvals;
    plan->tiles.kvals = kvals;

    raijinPrivateAllocStuff<T>(plan->abufs,ctx,transA,optparams.transA,optparams.simdwidth,lda,imax,kmax,ivals,kvals,useImageA,msize);
    raijinPrivateAllocStuff<T>(plan->bbufs,ctx,transB,optparams.transB,optparams.simdwidth,ldb,kmax,jmax,kvals,jvals,useImageB,msize);
    return plan;

}

template <typename T>
bool raijinGemmExecCopy(RaijinGemmPlan *plan,
                        cl_command_queue q,
						RaijinCleaner *cleaner,
                        cl_mem bufA,
                        cl_mem bufB,
                        cl_mem bufC,
                        int ldc,
                        bool doFirst,bool doSecond,
                        cl_event *retevt,RaijinTranspose *transObj,RaijinCopy *copyObj,RaijinScale *scale){
    if(retevt!=NULL) *retevt = NULL;
    const int imax = plan->tiles.ivals.size()-1;
    const int jmax = plan->tiles.jvals.size()-1;
    const int kmax = plan->tiles.kvals.size()-1;
//    cl_event *waitList = params.num_events>0? params.waitEvents:NULL;
    //cl_uint numEvents = params.num_events;
    std::vector<cl_event> acopyevts(imax*kmax);
    std::vector<cl_event> bcopyevts(kmax*jmax);
    if(doFirst){
        for(int i=0;i<imax;i++){
            for(int k=0;k<kmax;k++){
                const int index = i*kmax+k;
                const bool isAlloc = plan->abufs.isAllocated[index];
                const bool isCopy = plan->abufs.isCopy[index];
                const bool isTrans = !isCopy;
                const cl_mem Anew = plan->abufs.memobj[index];
                int startRow,endRow,startCol,endCol;
                if(plan->transA){
                    startCol = plan->tiles.ivals[i];
                    endCol = plan->tiles.ivals[i+1];
                    startRow = plan->tiles.kvals[k];
                    endRow = plan->tiles.kvals[k+1];
                }else{
                    startRow = plan->tiles.ivals[i];
                    endRow = plan->tiles.ivals[i+1];
                    startCol = plan->tiles.kvals[k];
                    endCol = plan->tiles.kvals[k+1];
                }
                const int simd = plan->optparams.simdwidth;
                const int lda = plan->lda;
                cl_event evt = NULL;
                //std::cout<<"raijinGemmExecCopy: A isAlloc "<<isAlloc<<" isCopy "<<isCopy<<" isTrans "<<isTrans<<" isImg "<<plan->optparams.imageA<<std::endl;
                if(isAlloc && isCopy){
                    if(plan->optparams.imageA){
                        evt = raijinCopyToImg<T>(copyObj,q,bufA,Anew,simd,startRow,endRow,startCol,endCol,lda);
                    }else{
                        //std::cout<<"raijinExecGemmCopy: Dispatching A copy "<<startRow<<" "<<" "<<endRow<<" "<<startCol<<" "<<endCol<<std::endl;
                        evt = raijinCopyToBuf<T>(copyObj,q,bufA,Anew,startRow,endRow,startCol,endCol,lda);
                    }
                }else if(isAlloc && isTrans){
                    if(plan->optparams.imageA){
                        evt = transToImg<T>(transObj,q,bufA,Anew,simd,startRow,endRow,startCol,endCol,lda);
                    }else{
						//std::cout<<"raijinExecGemmCopy: Dispatching A trans to buf "<<startRow<<" "<<" "<<endRow<<" "<<startCol<<" "<<endCol<<std::endl;
                        evt = transToBuf<T>(transObj,q,bufA,Anew,simd,startRow,endRow,startCol,endCol,lda);
                    }

                }else{
                    plan->abufs.memobj[index] = bufA;
                }
                acopyevts[index] = evt;
				if(evt!=NULL) *retevt = evt;
            }
        }
    }
    if(doSecond){
        for(int k=0;k<kmax;k++){
            for(int j=0;j<jmax;j++){
                const int index = k*jmax+j;
                const bool isAlloc = plan->bbufs.isAllocated[index];
                const bool isCopy = plan->bbufs.isCopy[index];
                const bool isTrans = !isCopy;
                const cl_mem Bnew = plan->bbufs.memobj[index];
                int startRow,endRow,startCol,endCol;
                if(plan->transB){
                    startCol = plan->tiles.kvals[k];
                    endCol = plan->tiles.kvals[k+1];
                    startRow = plan->tiles.jvals[j];
                    endRow = plan->tiles.jvals[j+1];
                }else{
                    startRow = plan->tiles.kvals[k];
                    endRow = plan->tiles.kvals[k+1];
                    startCol = plan->tiles.jvals[j];
                    endCol = plan->tiles.jvals[j+1];
                }
                const int simd = plan->optparams.simdwidth;
                const int ldb = plan->ldb;
                cl_event evt = NULL;
      
                //std::cout<<"raijinGemmExecCopy: B isAlloc "<<isAlloc<<" isCopy "<<isCopy<<" isTrans "<<isTrans<<" isImg "<<plan->optparams.imageB<<std::endl;
                if(isAlloc && isCopy){
                    if(plan->optparams.imageB){
                        evt = raijinCopyToImg<T>(copyObj,q,bufB,Bnew,simd,startRow,endRow,startCol,endCol,ldb);

                    }else{
                        //std::cout<<"raijinExecGemmCopy: Dispatching B copy "<<startRow<<" "<<" "<<endRow<<" "<<startCol<<" "<<endCol<<std::endl;
                        evt = raijinCopyToBuf<T>(copyObj,q,bufB,Bnew,startRow,endRow,startCol,endCol,ldb);
                    }
                }else if(isAlloc && isTrans){
                    if(plan->optparams.imageB){
                        evt = transToImg<T>(transObj,q,bufB,Bnew,simd,startRow,endRow,startCol,endCol,ldb);
                    }else{
                        //std::cout<<"raijinExecGemmCopy: Dispatching B trans "<<startRow<<" "<<" "<<endRow<<" "<<startCol<<" "<<endCol<<std::endl;
                        evt = transToBuf<T>(transObj,q,bufB,Bnew,simd,startRow,endRow,startCol,endCol,ldb);
                    }

                }else{
                    plan->bbufs.memobj[index] = bufB;
                }
                bcopyevts[index] = evt;
				if(evt!=NULL) *retevt = evt;
            }
        }
    }

    return true;
}

template <typename T>
T raijinGetOne(){ return 1;}

template <>
cl_double2 raijinGetOne();

template <>
cl_float2 raijinGetOne();


template <typename T>
cl_event raijinApplyOpt(cl_command_queue q, RaijinCleaner *cleaner,cl_kernel krnl,
                        RaijinGemmOptKernel optparams,
                        cl_context ctx,
                        cl_device_id dvc,
                        enum RAIJIN_ORDER order,
                        bool transA,
                        bool transB,
                        cl_uint M,
                        cl_uint N,
                        cl_uint K,
                        T alpha,
                        cl_mem A,
                        cl_uint lda,
                        cl_mem B,
                        cl_uint ldb,
                        T beta,
                        cl_mem C,
                        cl_uint ldc,
                        RaijinTranspose *transObj,RaijinCopy *copyObj,RaijinScale *scaleObj){
	using namespace std;
                           //cout<<"Inside applyOpt"<<endl;
    const size_t elemsize = sizeof(T);
    cl_device_type dvctype;
    clGetDeviceInfo(dvc,CL_DEVICE_TYPE,sizeof(dvctype),&dvctype,NULL);
    const int msize = (dvctype==CL_DEVICE_TYPE_CPU)? 1024:10000;
    
	cl_event scaleEvt = raijinScale<T>(scaleObj,q,C,M,N,ldc,beta);

	RaijinGemmPlan *plan = raijinGetGemmPlan<T>(optparams,ctx,dvc,order,transA,transB,M,N,K,lda,ldb,msize);
	cleaner->plan = plan;
    cl_event copyEvt=NULL;
    raijinGemmExecCopy<T>(plan,q,cleaner,A,B,C,ldc,true,true,&copyEvt,transObj,copyObj,scaleObj);
	//clFlush(q);
    const int imax = plan->tiles.ivals.size()-1;
    const int jmax = plan->tiles.jvals.size()-1;
    const int kmax = plan->tiles.kvals.size()-1;

	/*if(copyEvt!=NULL){
        //cout<<"raijinApplyOpt: Waitliset set to copy"<<endl;
        waitList = &copyEvt;
		numEvents = 1;
		curEventIndex = 0;
    }else{
		waitList = params.waitEvents;
		numEvents = params.num_events;
	}*/
	//clFinish(q);
	//RTimer rt;
	//rt.start();
	cl_event lastEvt;
    for(int i=0;i<imax;i++){
        for(int j=0;j<jmax;j++){
            size_t gsize[2], lsize[2];
            lsize[1] = optparams.lsizex;
            lsize[0] = optparams.lsizey;
            const cl_uint Mnew = plan->tiles.ivals[i+1]-plan->tiles.ivals[i];
            gsize[1] = Mnew/optparams.htile;
            const cl_uint Nnew = plan->tiles.jvals[j+1]-plan->tiles.jvals[j];
            gsize[0] = Nnew/optparams.wtile;
            cl_buffer_region region;
            region.origin = (i*ldc*msize + j*msize)*sizeof(T);
            region.size = (Mnew*ldc - j*msize)*sizeof(T);
            //cout<<"C origin: ("<<(region.origin/(ldc*sizeof(cl_float)))<<","<<(region.origin%(ldc*sizeof(cl_float)))/sizeof(cl_float)<<")"<<endl;
            cl_mem Cnew;
            Cnew = clCreateSubBuffer(C,CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&region,NULL);
            for(int k=0;k<kmax;k++){
                //cout<<"applyRegionOpt iteration ("<<i<<","<<j<<","<<k<<")"<<endl;
                //cout<<"Size ("<<Mnew<<","<<Nnew<<")"<<endl;
                cl_int kcode0, kcode1, kcode2, kcode3, kcode4, kcode5, kcode6, kcode7,kcode8;
                const cl_uint Kd = plan->tiles.kvals[k+1]-plan->tiles.kvals[k];
                //int prevEventIndex = curEventIndex-1;
                //cout<<"applyOpt: "<<Mnew<<" "<<Nnew<<" "<<Kd<<endl;
                const cl_mem Anew = plan->abufs.memobj[i*kmax+k];
                const cl_mem Bnew = plan->bbufs.memobj[k*jmax+j];
                /*size_t Asize,Bsize;
                clGetMemObjectInfo(Anew,CL_MEM_SIZE,sizeof(size_t),&Asize,NULL);
                clGetMemObjectInfo(Bnew,CL_MEM_SIZE,sizeof(size_t),&Bsize,NULL);
                cout<<"Size (num elements) of A "<<(Asize/sizeof(T))<<" Required "<<(Mnew*Kd)<<endl;
                cout<<"Size (num elements) of B "<<(Bsize/sizeof(T))<<" Required "<<(Nnew*Kd)<<endl;*/
                kcode0 = clSetKernelArg(krnl, 0, sizeof(cl_mem), &Anew);
                kcode1 = clSetKernelArg(krnl, 1, sizeof(cl_mem), &Bnew);
                kcode2 = clSetKernelArg(krnl, 2, sizeof(cl_mem), &Cnew);
				if(plan->abufs.isAllocated[i*kmax+k]){
					if(!optparams.transA){
						kcode3 = clSetKernelArg(krnl, 3, sizeof(cl_uint), &Kd);
					}else{
						kcode3 = clSetKernelArg(krnl,3,sizeof(cl_uint),&Mnew);
					}
				}else{
					//cout<<"Setting lda to "<<lda<<endl;
					kcode3 = clSetKernelArg(krnl,3,sizeof(cl_uint),&lda);
				}

				if(plan->bbufs.isAllocated[k*jmax+j]){
					if(!optparams.transB){
						kcode4 = clSetKernelArg(krnl, 4, sizeof(cl_uint), &Nnew);
					}else{
						kcode4 = clSetKernelArg(krnl, 4, sizeof(cl_uint), &Kd);
					}
				}else{
					//cout<<"Setting ldb to "<<ldb<<endl;
					kcode4 = clSetKernelArg(krnl,4,sizeof(cl_uint),&ldb);
				}

                kcode5 = clSetKernelArg(krnl, 5, sizeof(cl_uint), &ldc);
                kcode6 = clSetKernelArg(krnl, 6, sizeof(cl_uint), &Kd);
                kcode7 = clSetKernelArg(krnl, 7, elemsize, &alpha);
                T newbeta = raijinGetOne<T>();
                //if(k==0) newbeta = beta;
                kcode8 = clSetKernelArg(krnl, 8, elemsize, &beta);
				cl_event evt;
               // cout<<"Dispatching "<<gsize[1]<<" "<<gsize[0]<<" "<<lsize[1]<<" "<<lsize[0]<<endl;
				//system("PAUSE");
                cl_int errcode = clEnqueueNDRangeKernel(q, krnl, 2,NULL, gsize, lsize,0, NULL, &evt);
				lastEvt = evt;
				if(errcode!=CL_SUCCESS){
					std::cout<<"applyRegionOpt: "<<kcode0<<" "<<kcode1<<" "<<kcode2<<" "<<kcode3<<" "<<kcode4<<" "<<kcode5<<" "<<kcode6<<" "<<kcode7<<" "<<kcode8<<std::endl;
					std::cout<<"applyRegionOpt: return code "<<errcode<<std::endl;
				}
                //clWaitForEvents(1,&events[curEventIndex]);
                //cout<<"Finished dispatching this iteration"<<endl;
                //curEventIndex++;
            }
            //cout<<curEventIndex<<endl;

			//TODO: Need to put these on a resource cleaner somewhere?
			cleaner->bufs.push_back(Cnew);
            //clSetEventCallback(events[curEventIndex-1],CL_COMPLETE,releaseMemObject,(void *)Cnewcopy);
            //clReleaseMemObject(Cnew);
        }
    }
	//clFinish(q);
	//rt.stop();
	//std::cout<<"applyOpt: Time to finish kernels "<<rt.getDiff()<<std::endl;
    return lastEvt;

}

template <typename T>
class RaijinGemm{
public:
    static RaijinGemm<T> *getInstance(cl_context ctx,cl_device_id dvc);
    cl_event apply(cl_command_queue q, RaijinCleaner **cleaner, enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            T alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
            cl_mem B, cl_uint ldb, cl_uint offsetB,
            T beta,
            cl_mem C, cl_uint ldc, cl_uint offsetC);
    ~RaijinGemm();
private:
    RaijinTranspose *transObj;
    RaijinScale *scaleObj;
    RaijinCopy *copyObj;
    RaijinGemm();
    RaijinGemm(const RaijinGemm&);
    cl_event applyGen(cl_command_queue q,RaijinCleaner *cleaner, enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            T alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
            cl_mem B, cl_uint ldb, cl_uint offsetB,
            T beta,
            cl_mem C, cl_uint ldc, cl_uint offsetC);
    cl_event applyOpt( cl_command_queue q, RaijinCleaner *cleaner,enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
            T alpha, cl_mem A, cl_uint lda,
            cl_mem B, cl_uint ldb,
            T beta,
            cl_mem C, cl_uint ldc);
    RaijinGemmOptKernel optkernel;
    cl_context ctx;
    cl_device_id dvc;
    cl_kernel optcompiled;
    cl_kernel gencompiled;
    cl_program optprg;
    cl_program genprg;
};

template <typename T>
std::string getGemmname();

template <>
std::string getGemmname<cl_float>();

template <>
std::string getGemmname<cl_float2>();

template <>
std::string getGemmname<cl_double>();

template <>
std::string getGemmname<cl_double2>();

template <typename T>
std::string getGenGemmKernel();

template <>
std::string getGenGemmKernel<cl_float>();

template <>
std::string getGenGemmKernel<cl_float2>();

template <>
std::string getGenGemmKernel<cl_double>();

template <>
std::string getGenGemmKernel<cl_double2>();

template <typename T>
RaijinGemm<T> *RaijinGemm<T>::getInstance(cl_context context,cl_device_id device){
    std::string dpath = raijinGetProfileFileName(device,getGemmname<T>());
    std::cout<<"Filename "<<dpath<<std::endl;
    std::string line;
    std::ifstream ifile(dpath.c_str());
    //std::cout<<"Opened file? "<<ifile.is_open()<<" Is Good? "<<ifile.good()<<std::endl;
    RaijinGemmOptKernel opts;
    bool foundProfile = false;
    if(ifile.good() && ifile.is_open()) foundProfile = true;
    if(foundProfile){
        //std::cout<<"RaijinCL: Found the device profile"<<std::endl;
        RaijinGemm<T> *gemm = new RaijinGemm<T>;
		ifile>>gemm->optkernel;
        gemm->ctx = context;
        clRetainContext(context);
        gemm->dvc = device;
        cl_int errcode;
        const size_t len = gemm->optkernel.kernel.length();
        const char *prgstr = gemm->optkernel.kernel.c_str();
        //cout<<"Kernel:"<<gemm->optkernel.kernel<<endl;

        //TODO: check that there were no errors
        gemm->optprg = clCreateProgramWithSource(gemm->ctx, 1, &prgstr, &len, &errcode);
        //cout<<"Create program from source "<<errcode<<endl;
        cl_int bldcode = clBuildProgram(gemm->optprg, 1, &(gemm->dvc), "-cl-mad-enable", NULL, NULL);
        //cout<<"Build code "<<bldcode<<endl;
        gemm->optcompiled = clCreateKernel(gemm->optprg, gemm->optkernel.kname.c_str(), &errcode);
        std::string genKernel = getGenGemmKernel<T>();
        const char *genString = genKernel.c_str();
        const size_t genLength = genKernel.length();
        gemm->genprg = clCreateProgramWithSource(gemm->ctx, 1, &genString, &genLength, &errcode);
        bldcode = clBuildProgram(gemm->genprg, 1, &(gemm->dvc), "", NULL, NULL);
        std::string kname = getGemmname<T>();
        gemm->gencompiled = clCreateKernel(gemm->genprg, kname.c_str() , &errcode);
        gemm->transObj = new RaijinTranspose(device,context);
        gemm->copyObj = new RaijinCopy(context,device);
        gemm->scaleObj = new RaijinScale(context,device);
        return gemm;
    }else{
        std::cout<<"Did not find the profile"<<std::endl;
    }
    return NULL;
}

template <typename T>
cl_event RaijinGemm<T>::applyGen(cl_command_queue q, RaijinCleaner *cleaner,enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        T alpha, cl_mem A, cl_uint lda, unsigned int offsetA,
        cl_mem B, cl_uint ldb, unsigned int offsetB,
        T beta,
        cl_mem C, cl_uint ldc, unsigned int offsetC){
    //cout<<"Entering applyGen "<<M<<" "<<N<<" "<<K<<endl;
    const size_t elemsize = sizeof(T);
    cl_kernel krnl = gencompiled;
    cl_int kcode0, kcode1, kcode2, kcode3, kcode4, kcode5, kcode6, kcode7,kcode8,kcode9,kcode10,kcode11;
    cl_mem opA = A;
    if(transA){
        //cl_event transToBuf(RaijinTranspose *trans,
        //                    RaijinParams rp,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda);
       opA = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(M*K*sizeof(T)),NULL,NULL);
       cl_buffer_region region;
       region.origin = offsetA;
       region.size = ((K-1)*lda + M)*sizeof(T);
       cl_mem subBufferA = clCreateSubBuffer(A,CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION,&region,NULL);
       transToBuf<T>(transObj,q,subBufferA,opA,1,0,K,0,M,lda);
       lda = K;
    }

    cl_mem opB = B;
    if(transB){
        /*B is transposed. We want to convert it to a matrix of shape K*N */
        opB = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizeof(K*N*sizeof(T)),NULL,NULL);
        cl_buffer_region region;
        region.origin = offsetB;
        region.size = ((N-1)*lda + K)*sizeof(T);
        cl_mem subBufferB = clCreateSubBuffer(B,CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION,&region,NULL);
        transToBuf<T>(transObj,q,subBufferB,opB,1,0,N,0,K,ldb);
        ldb = K;
    }
    kcode0 = clSetKernelArg(krnl, 0, sizeof(cl_uint), &K);
    kcode1 = clSetKernelArg(krnl, 1, elemsize, &alpha);
    kcode2 = clSetKernelArg(krnl, 2, sizeof(cl_mem), &opA);
    kcode3 = clSetKernelArg(krnl, 3, sizeof(cl_uint), &lda);
    kcode4 = clSetKernelArg(krnl, 4, sizeof(cl_uint), &offsetA);
    kcode5 = clSetKernelArg(krnl, 5, sizeof(cl_mem), &B);
    kcode6 = clSetKernelArg(krnl, 6, sizeof(cl_uint), &ldb);
    kcode7 = clSetKernelArg(krnl, 7, sizeof(cl_uint), &opB);
    kcode8 = clSetKernelArg(krnl, 8, elemsize, &beta);
    kcode9 = clSetKernelArg(krnl, 9, sizeof(cl_mem), &C);
    kcode10 = clSetKernelArg(krnl, 10, sizeof(cl_uint), &ldc);
    kcode11 = clSetKernelArg(krnl, 11, sizeof(cl_uint), &offsetC);
    //cout<<kcode0<<" "<<kcode1<<" "<<kcode2<<" "<<kcode3<<" "<<kcode4<<" "<<kcode5<<" "<<kcode6<<" "<<kcode7<<" "<<kcode8<<" "<<kcode9<<" "<<kcode10<<
    //		" "<<kcode11<<endl;
    size_t gsize[2], lsize[2];
    gsize[1] = M;
    gsize[0] = N ;
    cl_event evt;
    cl_int errcode = clEnqueueNDRangeKernel(q, krnl, 2, NULL, gsize, NULL, 0, NULL, &evt);
    //cout<<"Was CL_SUCCESS? "<<(errcode==CL_SUCCESS)<<" "<<errcode<<endl;
    return evt;
}

template <typename T>
cl_event RaijinGemm<T>::apply(cl_command_queue q,RaijinCleaner **cleaner,enum RAIJIN_ORDER order, bool transA, bool transB,  cl_uint M, cl_uint N, cl_uint K,
        T alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
        cl_mem B, cl_uint ldb, cl_uint offsetB,
        T beta,
        cl_mem C, cl_uint ldc, cl_uint offsetC){
    //cout<<"Entering apply"<<endl;
    const int elemsize = sizeof(T);
    if(order==RaijinColMajor){
        //untested
        return apply(q,cleaner,RaijinRowMajor,transA,transB,N,M,K,alpha,B,ldb,offsetB,A,lda,offsetA,beta,C,ldc,offsetC);
    }

	*cleaner = new RaijinCleaner;
    cl_mem optA,optB,optC;
    int optM,optN,optK;
    optM = M - M%(optkernel.htile*optkernel.lsizex);
    optK = K - K%(optkernel.ktile);
    optN = N - N%(optkernel.wtile*optkernel.lsizey);
	cl_event lastEvt;
    //cout<<optM<<" "<<optN<<" "<<optK<<" "<<optkernel.ktile<<endl;
    cl_int errcode;
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

        lastEvt = applyOpt(q,*cleaner,order,transA,transB,optM,optN,optK,alpha,optA,lda,optB,ldb,beta,optC,ldc);
        if(offsetA>0) clReleaseMemObject(optA);
        if(offsetB>0) clReleaseMemObject(optB);
        if(offsetC>0) clReleaseMemObject(optC);
    }

    //calculate remaining summations for optM*optN portion of C
    if(K>optK){
        //cout<<"Remaining K"<<endl;
        int remK = K - optK;
        lastEvt = applyGen(q,*cleaner,order,transA,transB,optM,optN,remK,alpha,A,lda, offsetA+optK, B,ldb,offsetB+optK*ldb, beta, C,ldc,offsetC);
    }

    //Calculate the (M-optM) remaining rows of C
    if(M>optM){
        //cout<<"Remaining M"<<endl;
        int remM = M- optM;
        lastEvt = applyGen(q,*cleaner,order,transA,transB,remM,N,K,alpha, A,lda, offsetA+optM*lda, B,ldb,offsetB, beta, C,ldc,offsetC+optM*ldc);
    }

    if(N>optN){
        //cout<<"Remaining N"<<endl;
        int remN = N- optN;
        lastEvt = applyGen(q,*cleaner,order,transA,transB, M,remN,K,alpha, A,lda, offsetA, B,ldb,offsetB+optN, beta, C,ldc,offsetC+optN);
    }
    return lastEvt;
}

template <typename T>
RaijinGemm<T>::RaijinGemm(){
}

template <typename T>
RaijinGemm<T>::~RaijinGemm(){
    delete transObj;
    delete copyObj;
    delete scaleObj;
    clReleaseKernel(optcompiled);
    clReleaseKernel(gencompiled);
    clReleaseProgram(optprg);
    clReleaseProgram(genprg);
    clReleaseContext(ctx);
	
}

template <typename T>
cl_event RaijinGemm<T>::applyOpt(cl_command_queue q, RaijinCleaner *cleaner, enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        T alpha, cl_mem A, cl_uint lda,
        cl_mem B, cl_uint ldb,
        T beta,
        cl_mem C, cl_uint ldc){
    //cout<<"Entering applyOpt "<<M<<" "<<N<<" "<<K<<endl;
    return raijinApplyOpt<T>(q,cleaner,optcompiled,optkernel,ctx,dvc,
                          order,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,transObj,copyObj,scaleObj);

}

typedef RaijinGemm<cl_float> RaijinSgemm;
typedef RaijinGemm<cl_double> RaijinDgemm;


bool supportsFp64(cl_device_id dvc);
bool isAmd64(cl_device_id dvc);


}
#endif

