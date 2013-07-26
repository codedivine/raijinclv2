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
using namespace RaijinCL;
using namespace std;

static const string sgemmGenKernel = "__kernel void sgemmGen(int K, float alpha, "
		"const float __global *A, int lda, unsigned int offsetA, "
		"const float __global *B, int ldb, unsigned int offsetB, "
		"float beta,"
		" __global float *C, int ldc, unsigned int offsetC){\n"
		" int i = get_global_id(1);\n"
		" int j = get_global_id(0);\n"
		" int k;\n"
		" float sum = 0;"
		" for(k=0;k<K;k++){"
		"	sum += A[i*lda+k+offsetA]*B[k*ldb + j+offsetB];"
		"}"
		" C[i*ldc + j+offsetC] = alpha*sum + beta*C[i*ldc+j];"
		"}";

static const string dgemmGenKernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "__kernel void dgemmGen(int K, double alpha, "
		"const double __global *A, unsigned int lda, int offsetA, "
		"const double __global *B, unsigned int ldb, int offsetB, "
		"double beta,"
		" __global double *C, unsigned int ldc, int offsetC){\n"
		" int i = get_global_id(1);\n"
		" int j = get_global_id(0);\n"
		" int k;\n"
		" double sum = 0;"
		" for(k=0;k<K;k++){"
		"	sum += A[i*lda+k+offsetA]*B[k*ldb + j+offsetB];"
		"}"
		" C[i*ldc + j+offsetC] = alpha*sum + beta*C[i*ldc+j];"
		"}";


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
		}/*else if(ext.compare("cl_amd_fp64")==0){
			supports = true;
			break;
		}*/
    }
    delete[] extensions;
    return supports;
}

string RaijinCL::raijinGetProfileFileName(cl_device_id dvc, string prefix){
	string dpath = getenv("RAIJIN_TUNE_PATH");
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
	dpath = dpath + fname;
	return dpath;
}


istream& RaijinCL::operator>>(istream &stream,RaijinGemmOptKernel& krnl){
	string line;
	//Must be "start_opt_kernel"
	getline(stream,line);
	getline(stream,krnl.dvcname);
	//cout<<"Found device name "<<krnl.dvcname<<endl;
	stream>>krnl.lsizex;
	stream>>krnl.lsizey;
	stream>>krnl.htile;
	stream>>krnl.wtile;
	stream>>krnl.ktile;
	stream>>krnl.simdwidth;
    //cout<<"operator>>: Now reading orientation"<<endl;
    //Must be "orientation"
    stream>>line;
    stream>>krnl.transA;
    stream>>krnl.transB;

	//must be isImage
	stream>>line;
	stream>>krnl.imageA;
	stream>>krnl.imageB;
    //cout<<"Read orientation: "<<line<<" "<<krnl.transA<<" "<<krnl.transB<<endl;

    stream>>krnl.kname;
	//cout<<"Kernel name "<<krnl.kname<<endl;
	//Must be "start_kernel"
	stream>>line;
	//cout<<"start_kernel "<<line<<endl;
	stringstream kstream;
	while(true){
		getline(stream,line);
		if(line.compare("end_kernel")==0) break;
		kstream<<line<<" "<<endl;
	}
	krnl.kernel = kstream.str();
	//Must be "end_opt_kernel"
	getline(stream,line);
	return stream;
}

template<>
bool RaijinCL::raijinIsDouble<cl_double2>(){
    return true;
}

template<>
bool RaijinCL::raijinIsDouble<cl_double>(){
    return true;
}


ostream& RaijinCL::operator<<(ostream &stream,RaijinGemmOptKernel krnl){
	stream<<"start_opt_kernel"<<endl;
	stream<<krnl.dvcname<<endl;
	stream<<krnl.lsizex<<" "<<krnl.lsizey<<" "<<krnl.htile<<" "<<krnl.wtile<<" "<<krnl.ktile<<" "<<krnl.simdwidth<<endl;
    stream<<"orientation "<<krnl.transA<<" "<<krnl.transB<<endl;
	stream<<"isImage "<<krnl.imageA<<" "<<krnl.imageB<<endl;
	stream<<krnl.kname<<endl;
	stream<<"start_kernel"<<endl;
	stream<<krnl.kernel<<endl;
	stream<<"end_kernel"<<endl;
	stream<<"end_opt_kernel"<<endl;
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
