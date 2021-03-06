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

#include "raijin.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <CL/cl.h>
#include "rtimer.hpp"
#include "json/json.h"
using namespace RaijinCL;
using namespace std;

#ifdef RAIJIN_EXPERIMENTAL
#ifdef RAIJIN_AMD
#pragma comment(lib,"libacml_mp_dll")
#endif
#endif

static const string sgemmGenKernel = "__kernel void sgemmGen(int K, float alpha, "
        "const float __global *A, int sa0, int sa1, unsigned int offsetA, "
        "const float __global *B, int sb0, int sb1, unsigned int offsetB, "
		"float beta,"
		" __global float *C, int ldc, unsigned int offsetC){\n"
		" int i = get_global_id(1);\n"
		" int j = get_global_id(0);\n"
		" int k;\n"
		" float sum = 0;"
		" for(k=0;k<K;k++){"
        "	sum += A[i*sa0 + k*sa1 +offsetA]*B[j*sb0 + k*sb1 +offsetB];"
		"}"
		" C[i*ldc + j+offsetC] = alpha*sum + beta*C[i*ldc+j];"
		"}";

static const string dgemmGenKernel = "#ifdef cl_khr_fp64\n"
		"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
		"#else\n"
		"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
        "#endif\n"
        "__kernel void dgemmGen(int K, double alpha, "
        "const double __global *A, int sa0, int sa1,int offsetA, "
        "const double __global *B, int sb0, int sb1,int offsetB, "
		"double beta,"
		" __global double *C, unsigned int ldc, int offsetC){\n"
		" int i = get_global_id(1);\n"
		" int j = get_global_id(0);\n"
		" int k;\n"
		" double sum = 0;"
		" for(k=0;k<K;k++){"
        "	sum += A[i*sa0 + k*sa1 +offsetA]*B[j*sb0 + k*sb1 +offsetB];"
		"}"
		" C[i*ldc + j+offsetC] = alpha*sum + beta*C[i*ldc+j];"
		"}";

std::ostream& RaijinCL::operator<<(std::ostream& str,cl_double2 v){
    str<<'('<<v.s[0]<<','<<v.s[1]<<')';
    return str;
}

std::ostream& RaijinCL::operator<<(std::ostream& str,cl_float2 v){
    str<<'('<<v.s[0]<<','<<v.s[1]<<')';
    return str;
}

bool RaijinCL::supportsFp64(cl_device_id dvc){
    const int size = 10000;
    char *extensions = new char[size];
    clGetDeviceInfo(dvc,CL_DEVICE_EXTENSIONS,size,extensions,NULL);
    string extstring(extensions);
    stringstream extstream(extstring);
    string ext;
    bool supports = false;
    while(extstream.good()){
        extstream>>ext;
        if(ext.compare("cl_khr_fp64")==0){
            supports = true;
            break;
		}else if(ext.compare("cl_amd_fp64")==0){
			supports = true;
			break;
		}
    }
    delete[] extensions;
    return supports;
}

bool RaijinCL::isAmd64(cl_device_id dvc){
    const int size = 10000;
    char *extensions = new char[size];
    clGetDeviceInfo(dvc,CL_DEVICE_EXTENSIONS,size,extensions,NULL);
    
	string extstring(extensions);

    stringstream extstream(extstring);
    string ext;
    bool supportsAmd64 = false;
	bool supportsKhr64 = false;
    while(extstream.good()){
        extstream>>ext;
        if(ext.compare("cl_amd_fp64")==0){
            supportsAmd64 = true;
		}
		if(ext.compare("cl_khr_fp64")==0){
			supportsKhr64 = true;
		}
    }

    delete[] extensions;
    if(!supportsAmd64) return false;
	if(supportsAmd64 && supportsKhr64) return false;
	return true;
}

string RaijinCL::raijinGetProfileFileName(cl_device_id dvc, string prefix){
    char *dpathPtr = getenv("RAIJIN_TUNE_PATH");
    string dpath( dpathPtr==NULL ? "" : dpathPtr);
	cl_uint vendorid;
	const size_t sizename = 100;
	char devname[sizename];
	clGetDeviceInfo(dvc,CL_DEVICE_NAME,sizeof(devname),devname,NULL);
	char modname[sizename];

	size_t idx = 0;
	bool prevWasUnderscore = true;
	for(size_t i=0;i<sizename;i++){
		char c = devname[i];
		if(c=='\0'){
			modname[idx] = '\0';
			break;
		}
		if(! (('a'<=c && c<='z') || ('A'<=c && c<='Z') || ('0'<=c && c<='9'))){
			if(prevWasUnderscore) continue;
			modname[idx] = '_';
			prevWasUnderscore = true;
		}else{
			modname[idx] = devname[i];
			prevWasUnderscore = false;
		}
		idx++;
	}
	clGetDeviceInfo(dvc,CL_DEVICE_VENDOR_ID,sizeof(cl_uint),&vendorid,NULL);
	stringstream fnamestream;
	fnamestream<<vendorid<<"_"<<modname;
    //cout<<fnamestream.str()<<endl;
    string fname = prefix+fnamestream.str();
    dpath = dpath + fname + ".json";
	return dpath;
}


template<>
bool RaijinCL::raijinIsDouble<cl_double2>(){
    return true;
}

template<>
bool RaijinCL::raijinIsDouble<cl_double>(){
    return true;
}

ostream& RaijinCL::operator<<(ostream& stream,const RaijinGemmOptKernel& krnl){
    Json::Value root;
    Json::StyledWriter writer;
    root["dvcname"] = krnl.dvcname;
    root["lsizex"] = krnl.lsizex;
    root["lsizey"] = krnl.lsizey;
    root["htile"] = krnl.htile;
    root["wtile"] = krnl.wtile;
    root["ktile"] = krnl.ktile;
    root["simdwidth"] = krnl.simdwidth;
    root["transA"] = krnl.transA;
    root["transB"] = krnl.transB;
    root["imageA"] = krnl.imageA;
    root["imageB"] = krnl.imageB;
    root["kname"] = krnl.kname;
    root["kernel"] = krnl.kernel;
    stream<<writer.write(root)<<endl;
    return stream;
}

istream& RaijinCL::operator>>(istream &stream,RaijinGemmOptKernel& krnl){
    Json::Value root;
    stream>>root;
    krnl.dvcname = root.get("dvcname","").asString();
    krnl.lsizex = root.get("lsizex",0).asInt();
    krnl.lsizey = root.get("lsizey",0).asInt();
    krnl.htile = root.get("htile",0).asInt();
    krnl.wtile = root.get("wtile",0).asInt();
    krnl.ktile = root.get("ktile",0).asInt();
    krnl.simdwidth = root.get("simdwidth",0).asInt();
    krnl.transA = root.get("transA",false).asBool();
    krnl.transB = root.get("transB",false).asBool();
    krnl.imageA = root.get("imageA",false).asBool();
    krnl.imageB = root.get("imageB",false).asBool();
    krnl.kname = root.get("kname","").asString();
    krnl.kernel = root.get("kernel","").asString();
    return stream;
}

void RaijinCL::printProgramBinary(cl_program prg){
    //Assuming program was compiled for only 1 device
    size_t size;
    clGetProgramInfo(prg,CL_PROGRAM_BINARY_SIZES,sizeof(size),&size,NULL);
    unsigned char *binaries[1];
    cout<<"Binary size "<<size<<endl;
    binaries[0] = new unsigned char[size+1];
    size_t retbytes;
    clGetProgramInfo(prg,CL_PROGRAM_BINARIES,sizeof(unsigned char*),binaries,&retbytes);
    cout<<"Return bytes "<<retbytes<<endl;
    cout<<"Binary:"<<binaries[0]<<endl;

}

template <>
std::string RaijinCL::getGemmname<cl_float>(){
    return "sgemm";
}


template <>
std::string RaijinCL::getGemmname<cl_double>(){
    return "dgemm";
}

template <>
std::string RaijinCL::getGenGemmKernel<cl_float>(){
    return sgemmGenKernel;
}

template <>
std::string RaijinCL::getGenGemmKernel<cl_double>(){
    return dgemmGenKernel;
}



size_t RaijinCL::findMaxDivisor(size_t val,size_t max){
    while(true){
        if(val%max==0) return max;
        max = max/2;
    }
}

void RaijinCL::releaseMemObject(cl_event evt,cl_int status,void *data){
    //cout<<"Releasing mem object"<<endl;
    cl_mem* bufptr = (cl_mem*)data;
    cl_mem buf = *bufptr;
    /*cl_mem_object_type mtype;
    size_t msize;
    clGetMemObjectInfo(buf,CL_MEM_TYPE,sizeof(cl_mem_object_type),&mtype,NULL);
    clGetMemObjectInfo(buf,CL_MEM_SIZE,sizeof(size_t),&msize,NULL);
    if(mtype==CL_MEM_OBJECT_BUFFER) cout<<"Buffer of size "<<msize<<endl;
    else cout<<"Unknown memory object of type "<<msize<<endl;
    cout<<"Derefed pointer"<<endl;*/
    clReleaseMemObject(buf);
    //cout<<"Finished clrelease call"<<endl;
	delete bufptr;
    //cout<<"Released"<<endl;
}


/*
*/


RaijinGemmPlan::ExecBufs::~ExecBufs(){
    for(int i=0;i<memobj.size();i++){
        cl_mem buf = memobj[i];
        if(isAllocated[i]){
            //cout<<"Deleting buffer"<<endl;
            clReleaseMemObject(buf);
        }
    }
}

void RaijinGemmPlan::deleteGemmPlan(cl_event evt, cl_int status, void *vplan){
    //cout<<"Deleting plan"<<endl;
    RaijinGemmPlan *plan = (RaijinGemmPlan*)vplan;
    delete plan;
}

RaijinCleaner::~RaijinCleaner(){
	for(int i=0;i<bufs.size();i++) clReleaseMemObject(bufs[i]);
	if(plan!=NULL) delete plan;
}

#ifdef RAIJIN_EXPERIMENTAL
template <>
void BlasFun<float>(void* params){
	BlasParams<float> *myParams = (BlasParams<float>*)params;
#ifdef RAIJIN_AMD
	char transA = (myParams->transA) ? 'T' : 'N';
	char transB = (myParams->transB) ? 'T' : 'N';
	sgemm(transB,transA,myParams->N,myParams->M,myParams->K,myParams->alpha,myParams->B,
		myParams->ldb,myParams->A,myParams->lda,myParams->beta,myParams->C,myParams->ldc);
#endif
#ifdef RAIJIN_INTEL
	cblas_sgemm(CblasRowMajor,(myParams->transA)? CblasTrans : CblasNoTrans, (myParams->transB) ? CblasTrans : CblasNoTrans,
		myParams->M,myParams->N,myParams->K,myParams->alpha,myParams->A,myParams->lda,myParams->B,myParams->ldb,myParams->beta,
		myParams->C,myParams->ldc);
#endif

}

template<>
void BlasFun<double>(void* params){
	BlasParams<double> *myParams = (BlasParams<double>*)params;
#ifdef RAIJIN_AMD
	char transA = (myParams->transA == 'N') ? 'T' : 'N';
	char transB = (myParams->transB == 'N') ? 'N' : 'T';
	dgemm(transA,transB,myParams->N,myParams->M,myParams->K,myParams->alpha,myParams->B,
		myParams->ldb,myParams->A,myParams->lda,myParams->beta,myParams->C,myParams->ldc);
#endif
#ifdef RAIJIN_INTEL
	cblas_dgemm(CblasRowMajor,(myParams->transA)? CblasTrans : CblasNoTrans, (myParams->transB) ? CblasTrans : CblasNoTrans,
		myParams->M,myParams->N,myParams->K,myParams->alpha,myParams->A,myParams->lda,myParams->B,myParams->ldb,myParams->beta,
		myParams->C,myParams->ldc);
#endif
}
#endif
