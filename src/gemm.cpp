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
        }
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

RaijinSgemm::RaijinSgemm(){
}

RaijinSgemm::~RaijinSgemm(){
    delete transObj;
    delete copyObj;
    delete scaleObj;
    clReleaseKernel(optcompiled);
    clReleaseKernel(gencompiled);
    clReleaseProgram(optprg);
    clReleaseProgram(genprg);
    clReleaseContext(ctx);
}

RaijinSgemm *RaijinSgemm::getInstance(cl_context context,cl_device_id device){
    string dpath = raijinGetProfileFileName(device);
	//cout<<"Filename "<<dpath<<endl;
    string line;
    ifstream ifile(dpath.c_str());
	//cout<<"Opened file? "<<ifile.is_open()<<" Is Good? "<<ifile.good()<<endl;
	RaijinGemmOptKernel opts;
	bool foundProfile = false;
    while(true){
        if(!ifile.good()) break;
		ifile>>line;
        //cout<<"Read "<<line<<endl;
		if(line.compare("start_sgemm")==0){
            //cout<<"Reading!"<<endl;
			ifile>>opts;
			foundProfile= true;
			break;
		}
	}
	if(foundProfile){
		//cout<<"RaijinCL: Found the device profile"<<endl;
		RaijinSgemm *sgemm = new RaijinSgemm;
		sgemm->optkernel = opts;
		sgemm->ctx = context;
		clRetainContext(context);
		sgemm->dvc = device;
		cl_int errcode;
		const size_t len = sgemm->optkernel.kernel.length();
		const char *prgstr = sgemm->optkernel.kernel.c_str();
		//cout<<"Kernel:"<<sgemm->optkernel.kernel<<endl;

		//TODO: check that there were no errors
		sgemm->optprg = clCreateProgramWithSource(sgemm->ctx, 1, &prgstr, &len, &errcode);
		//cout<<"Create program from source "<<errcode<<endl;
		cl_int bldcode = clBuildProgram(sgemm->optprg, 1, &(sgemm->dvc), "", NULL, NULL);
		//cout<<"Build code "<<bldcode<<endl;
		sgemm->optcompiled = clCreateKernel(sgemm->optprg, sgemm->optkernel.kname.c_str(), &errcode);

		const char *genString = sgemmGenKernel.c_str();
		const size_t genLength = sgemmGenKernel.length();
		sgemm->genprg = clCreateProgramWithSource(sgemm->ctx, 1, &genString, &genLength, &errcode);
        bldcode = clBuildProgram(sgemm->genprg, 1, &(sgemm->dvc), "-g", NULL, NULL);
		sgemm->gencompiled = clCreateKernel(sgemm->genprg, "sgemmGen", &errcode);
        sgemm->transObj = new RaijinTranspose(device,context);
        sgemm->copyObj = new RaijinCopy(context,device);
        sgemm->scaleObj = new RaijinScale(context,device);
		return sgemm;
	}else{
		cout<<"Did not find the profile"<<endl;
	}
	return NULL;
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

cl_event RaijinSgemm::apply(enum RAIJIN_ORDER order, bool transA, bool transB,  cl_uint M, cl_uint N, cl_uint K,
        cl_float alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
        cl_mem B, cl_uint ldb, cl_uint offsetB,
        cl_float beta,
        cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params){
    //cout<<"Entering apply"<<endl;
    const int elemsize = sizeof(float);
    if(order==RaijinColMajor){
        //untested
        return apply(RaijinRowMajor,transA,transB,N,M,K,alpha,B,ldb,offsetB,A,lda,offsetA,beta,C,ldc,offsetC,params);
    }


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

    //calculate remaining summations for optM*optN portion of C
    if(K>optK){
        //cout<<"Remaining K"<<endl;
        int remK = K - optK;
        cl_event evt = applyGen(order,transA,transB,optM,optN,remK,alpha,A,lda, offsetA+optK, B,ldb,offsetB+optK*ldb, beta, C,ldc,offsetC, temp);
        temp.waitEvents[temp.num_events] = evt;
        temp.num_events++;
    }

    //Calculate the (M-optM) remaining rows of C
    if(M>optM){
        //cout<<"Remaining M"<<endl;
        int remM = M- optM;
        cl_event evt = applyGen(order,transA,transB,remM,N,K,alpha, A,lda, offsetA+optM*lda, B,ldb,offsetB, beta, C,ldc,offsetC+optM*ldc, temp);
        temp.waitEvents[temp.num_events] = evt;
        temp.num_events++;
    }

    if(N>optN){
        //cout<<"Remaining N"<<endl;
        int remN = N- optN;
        cl_event evt = applyGen(order,transA,transB, M,remN,K,alpha, A,lda, offsetA, B,ldb,offsetB+optN, beta, C,ldc,offsetC+optN, temp);
        temp.waitEvents[temp.num_events] = evt;
        temp.num_events++;
    }
    cl_event lastEvt = temp.waitEvents[temp.num_events-1];
    delete[] temp.waitEvents;
    return lastEvt;
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


cl_event RaijinSgemm::applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
		cl_float alpha, cl_mem A, cl_uint lda,
        cl_mem B, cl_uint ldb,
        cl_float beta,
        cl_mem C, cl_uint ldc, RaijinParams params){
    //cout<<"Entering applyOpt "<<M<<" "<<N<<" "<<K<<endl;
    return raijinApplyOpt<float>(optcompiled,optkernel,ctx,dvc,
                          order,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,params,transObj,copyObj,scaleObj);

}

cl_event RaijinSgemm::applyGen(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_float alpha, cl_mem A, cl_uint lda, unsigned int offsetA,
        cl_mem B, cl_uint ldb, unsigned int offsetB,
        cl_float beta,
        cl_mem C, cl_uint ldc, unsigned int offsetC, RaijinParams params){
    //cout<<"Entering applyGen "<<M<<" "<<N<<" "<<K<<endl;
    const size_t elemsize = sizeof(float);
    cl_kernel krnl = gencompiled;
    cl_int kcode0, kcode1, kcode2, kcode3, kcode4, kcode5, kcode6, kcode7,kcode8,kcode9,kcode10,kcode11;
    kcode0 = clSetKernelArg(krnl, 0, sizeof(cl_uint), &K);
    kcode1 = clSetKernelArg(krnl, 1, elemsize, &alpha);
    kcode2 = clSetKernelArg(krnl, 2, sizeof(cl_mem), &A);
    kcode3 = clSetKernelArg(krnl, 3, sizeof(cl_uint), &lda);
    kcode4 = clSetKernelArg(krnl, 4, sizeof(cl_uint), &offsetA);
    kcode5 = clSetKernelArg(krnl, 5, sizeof(cl_mem), &B);
    kcode6 = clSetKernelArg(krnl, 6, sizeof(cl_uint), &ldb);
    kcode7 = clSetKernelArg(krnl, 7, sizeof(cl_uint), &offsetB);
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
    cl_event *waitList = (params.num_events>0) ? params.waitEvents : NULL;
    cl_int errcode = clEnqueueNDRangeKernel(params.queue, krnl, 2, NULL, gsize, NULL, params.num_events, waitList, &evt);
    //cout<<"Was CL_SUCCESS? "<<(errcode==CL_SUCCESS)<<" "<<errcode<<endl;
    return evt;

}


RaijinDgemm::RaijinDgemm(){
}

RaijinDgemm::~RaijinDgemm(){
    delete transObj;
    delete copyObj;
    delete scaleObj;
    clReleaseKernel(optcompiled);
    clReleaseKernel(gencompiled);
    clReleaseProgram(optprg);
    clReleaseProgram(genprg);
    clReleaseContext(ctx);
}

RaijinDgemm *RaijinDgemm::getInstance(cl_context context,cl_device_id device){
	string dpath = raijinGetProfileFileName(device);
	//cout<<"Filename "<<dpath<<endl;
	string line;
	ifstream ifile(dpath.c_str());
	//cout<<"Opened file? "<<ifile.is_open()<<" Is Good? "<<ifile.good()<<endl;
	RaijinGemmOptKernel opts;
	bool foundProfile = false;
	while(true){
		if(!ifile.good()) break;
		ifile>>line;
		//cout<<"Read "<<line<<endl;
		if(line.compare("start_dgemm")==0){
			//cout<<"Reading!"<<endl;
			ifile>>opts;
			foundProfile= true;
			break;
		}
	}
	if(foundProfile){
		RaijinDgemm *dgemm = new RaijinDgemm;
		dgemm->optkernel = opts;
		dgemm->ctx = context;
		dgemm->dvc = device;
        clRetainContext(context);
		cl_int errcode;
		const size_t len = dgemm->optkernel.kernel.length();
		const char *prgstr = dgemm->optkernel.kernel.c_str();
		//cout<<"Kernel:"<<sgemm->optkernel.kernel<<endl;

		//TODO: check that there were no errors
		dgemm->optprg = clCreateProgramWithSource(dgemm->ctx, 1, &prgstr, &len, &errcode);
		//cout<<"Create program from source "<<errcode<<endl;
		cl_int bldcode = clBuildProgram(dgemm->optprg, 1, &(dgemm->dvc), "", NULL, NULL);
		//cout<<"Build code "<<bldcode<<endl;
		dgemm->optcompiled = clCreateKernel(dgemm->optprg, dgemm->optkernel.kname.c_str(), &errcode);
		const char *genString = dgemmGenKernel.c_str();
		const size_t genLength = dgemmGenKernel.length();
		dgemm->genprg = clCreateProgramWithSource(dgemm->ctx, 1, &genString, &genLength, &errcode);
		bldcode = clBuildProgram(dgemm->genprg, 1, &(dgemm->dvc), "", NULL, NULL);
		dgemm->gencompiled = clCreateKernel(dgemm->genprg, "dgemmGen", &errcode);
        dgemm->transObj = new RaijinTranspose(device,context);
        dgemm->copyObj = new RaijinCopy(context,device);
        dgemm->scaleObj = new RaijinScale(context,device);
		//TODO: build the generic kernel
		return dgemm;
	}else{
		cout<<"Did not find the profile"<<endl;
	}
	return NULL;
}

cl_event RaijinDgemm::applyGen(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_double alpha, cl_mem A, cl_uint lda, unsigned int offsetA,
        cl_mem B, cl_uint ldb, unsigned int offsetB,
        cl_double beta,
        cl_mem C, cl_uint ldc, unsigned int offsetC, RaijinParams params){
    //cout<<"Entering applyGen "<<M<<" "<<N<<" "<<K<<endl;
    const size_t elemsize = sizeof(double);
    cl_kernel krnl = gencompiled;
    cl_int kcode0, kcode1, kcode2, kcode3, kcode4, kcode5, kcode6, kcode7,kcode8,kcode9,kcode10,kcode11;
    kcode0 = clSetKernelArg(krnl, 0, sizeof(cl_uint), &K);
    kcode1 = clSetKernelArg(krnl, 1, elemsize, &alpha);
    kcode2 = clSetKernelArg(krnl, 2, sizeof(cl_mem), &A);
    kcode3 = clSetKernelArg(krnl, 3, sizeof(cl_uint), &lda);
    kcode4 = clSetKernelArg(krnl, 4, sizeof(cl_uint), &offsetA);
    kcode5 = clSetKernelArg(krnl, 5, sizeof(cl_mem), &B);
    kcode6 = clSetKernelArg(krnl, 6, sizeof(cl_uint), &ldb);
    kcode7 = clSetKernelArg(krnl, 7, sizeof(cl_uint), &offsetB);
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
    cl_event *waitList = (params.num_events>0) ? params.waitEvents : NULL;
    cl_int errcode = clEnqueueNDRangeKernel(params.queue, krnl, 2, NULL, gsize, NULL, params.num_events, waitList, &evt);
    //cout<<"Was CL_SUCCESS? "<<(errcode==CL_SUCCESS)<<" "<<errcode<<endl;
    return evt;

}

cl_event RaijinDgemm::applyOpt(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
        cl_double alpha, cl_mem A, cl_uint lda,
        cl_mem B, cl_uint ldb,
        cl_double beta,
        cl_mem C, cl_uint ldc, RaijinParams params){
    return raijinApplyOpt<double>(optcompiled,optkernel,ctx,dvc,
                          order,transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,params,transObj,copyObj,scaleObj);


}

cl_event RaijinDgemm::apply(enum RAIJIN_ORDER order, bool transA, bool transB, cl_uint M, cl_uint N, cl_uint K,
		cl_double alpha, cl_mem A, cl_uint lda, cl_uint offsetA,
		cl_mem B, cl_uint ldb, cl_uint offsetB,
		cl_double beta,
		cl_mem C, cl_uint ldc, cl_uint offsetC, RaijinParams params){
	const int elemsize = sizeof(double);

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

	//calculate remaining summations for optM*optN portion of C
	if(K>optK){
	//	cout<<"Remaining K"<<endl;
		int remK = K - optK;
		cl_event evt = applyGen(order,transA,transB,optM,optN,remK,alpha,A,lda, offsetA+optK, B,ldb,offsetB+optK*ldb, beta, C,ldc,offsetC, temp);
		temp.waitEvents[temp.num_events] = evt;
		temp.num_events++;
	}

	//Calculate the (M-optM) remaining rows of C
	if(M>optM){
	//	cout<<"Remaining M"<<endl;
		int remM = M- optM;
		cl_event evt = applyGen(order,transA,transB,remM,N,K,alpha, A,lda, offsetA+optM*lda, B,ldb,offsetB, beta, C,ldc,offsetC+optM*ldc, temp);
		temp.waitEvents[temp.num_events] = evt;
		temp.num_events++;
	}

	if(N>optN){
	//	cout<<"Remaining N"<<endl;
		int remN = N- optN;
		cl_event evt = applyGen(order,transA,transB, M,remN,K,alpha, A,lda, offsetA, B,ldb,offsetB+optN, beta, C,ldc,offsetC+optN, temp);
		temp.waitEvents[temp.num_events] = evt;
		temp.num_events++;
	}
	cl_event lastEvt = temp.waitEvents[temp.num_events-1];
	delete[] temp.waitEvents;
	return lastEvt;

}

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
