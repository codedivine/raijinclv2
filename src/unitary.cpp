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
#include <sstream>
using namespace RaijinCL;
using namespace std;

/*
__kernel void sqrtDoubleOneDim(__global const double *input, int stride, int ioffset, __global double *output, int ooffset){
    int i = get_global_id(0);
    output[ooffset + i] = sqrt(input[ioffset + i*stride]);
}

__kernel void sqrtDoubleTwoDim(__global const double *input, int s1,int s2, int ioffset, __global double *output, int ooffset,int n){
    int i = get_global_id(0);
    int j = get_global_id(1);
    output[ooffset + i*n + j] = sqrt(input[ioffset + i*s1 + j*s2]);
}

__kernel void sqrtDoubleMultiDim(__global const double *input,int ioffset, __global double *output, int ooffset, int ndims, __global const int *strides, __global const int *dims){
    int id = get_global_id(0);
    int curPos = id;
    int i;
    int inIndex = ioffset;
    int outIndex = ooffset;
    int multiplier = 1;
    for(i=ndims-1;i>=0;i--){
        int pos = curPos%dims[i];
        inIndex += pos*strides[i];
        outIndex += pos*multiplier;
        curPos = curPos/dims[i];
        multiplier *= dims[i];
    }
    output[outIndex] = sqrt(input[inIndex]);
}
*/

string RaijinUnitaryOp::getOpName(Op op){
    switch(op){
    case SQRT:
        return "sqrt";
    case LOG:
        return "log10";
    case LN:
        return "log";
    case EXP:
        return "exp";
    case SIN:
        return "sin";
    case COS:
        return "cos";
    case TAN:
        return "tan";
    }
}

string RaijinUnitaryOp::getOpSymbol(Op op){
    //for the defined ops, it is the same. Might change in future
    return RaijinUnitaryOp::getOpSymbol(op);
}

static string makeOneDimOp(RaijinDtypes::DType dtype, RaijinUnitaryOp::Op op){
    stringstream ss;
    const string dtypeName = RaijinDtypes::getDataTypeName(dtype);
    const string dtypeSym = RaijinDtypes::getDataTypeSym(dtype);
    const string opName = RaijinUnitaryOp::getOpName(op);
    const string opSym = RaijinUnitaryOp::getOpSymbol(op);
    if(RaijinDtypes::isDoubleBased(dtype)){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    ss<<"__kernel void "<<opName<<dtypeName<<"OneDim(";
    ss<<"__global const "<<dtypeSym<<" *input, int stride, int ioffset, __global "<<dtypeSym<<" *output,int ioffset){"<<endl;
    ss<<"int i = get_global_id(0);"<<endl;
    ss<<"output[ooffset + i] = "<<opSym<<"(input[ioffset + i*stride]);"<<endl;
    ss<<"}"<<endl;
    return ss.str();
}

static string makeTwoDimOp(RaijinDtypes::DType dtype, RaijinUnitaryOp::Op op){
    stringstream ss;
    const string dtypeName = RaijinDtypes::getDataTypeName(dtype);
    const string dtypeSym = RaijinDtypes::getDataTypeSym(dtype);
    const string opName = RaijinUnitaryOp::getOpName(op);
    const string opSym = RaijinUnitaryOp::getOpSymbol(op);
    if(RaijinDtypes::isDoubleBased(dtype)){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    ss<<"__kernel void "<<opName<<dtypeName<<"TwoDim(__global const "<<dtypeSym<<" *input, int s1,int s2, int ioffset, __global "
        <<dtypeSym<<" *output, int ooffset,int n){"<<endl;
    ss<<"int i = get_global_id(0);"<<endl;
    ss<<"int j = get_global_id(1);"<<endl;
    ss<<"output[ooffset + i*n + j] = "<<opSym<<"(input[ioffset + i*s1 + j*s2]);"<<endl;
    ss<<"}";
    return ss.str();
}

static string makeMultiDimOp(RaijinDtypes::DType dtype, RaijinUnitaryOp::Op op){
    stringstream ss;
    const string dtypeName = RaijinDtypes::getDataTypeName(dtype);
    const string dtypeSym = RaijinDtypes::getDataTypeSym(dtype);
    const string opName = RaijinUnitaryOp::getOpName(op);
    const string opSym = RaijinUnitaryOp::getOpSymbol(op);
    if(RaijinDtypes::isDoubleBased(dtype)){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    ss<<"__kernel void "<<opName<<dtypeName<<"MultiDim(__global const "<<dtypeSym<<" *input,int ioffset, __global "<<dtypeSym<<" *output, int ooffset,"
        <<"int ndims, __global const int *strides, __global const int *dims){"<<endl;
    ss<<"int id = get_global_id(0);"<<endl;
        ss<<"int curPos = id;"<<endl;
        ss<<"int i;"<<endl;
        ss<<"int inIndex = ioffset;"<<endl;
        ss<<"int outIndex = ooffset;"<<endl;
        ss<<"int multiplier = 1;"<<endl;
        ss<<"for(i=ndims-1;i>=0;i--){"
            <<"int pos = curPos%dims[i];"<<endl
            <<"inIndex += pos*strides[i];"<<endl
            <<"outIndex += pos*multiplier;"<<endl
            <<"curPos = curPos/dims[i];"<<endl
            <<"multiplier *= dims[i];"<<endl
            <<"}"<<endl
            <<"output[outIndex] = "<<opSym<<"(input[inIndex]);"<<endl
        <<"}"<<endl;
    return ss.str();
}

cl_event RaijinUnitaryOp::apply(Op op, RaijinDtypes::DType dtype, cl_mem input, cl_int ioffset, cl_mem output, cl_int ooffset, cl_int ndims, cl_int *dims, cl_int *strides){
    cl_event evt;
    const int nops = EXP - SQRT + 1;
    const int ndtypes = 3;
    const int nversions = 3;
    if(ndims==1){
        cl_kernel kernel = kernels[nversions*nops*dtype + nversions*op];
        cl_int stride = strides[0];
        clSetKernelArg(kernel,0,sizeof(cl_mem),&input);
        clSetKernelArg(kernel,1,sizeof(cl_int),&stride);
        clSetKernelArg(kernel,2,sizeof(cl_int),&ioffset);
        clSetKernelArg(kernel,3,sizeof(cl_mem),&output);
        clSetKernelArg(kernel,4,sizeof(cl_int),&ooffset);
        const size_t gsize = dims[0];
        clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,NULL,0,NULL,&evt);
    }else if(ndims==2){
          cl_kernel kernel = kernels[nversions*nops*dtype + nversions*op + 1];
          clSetKernelArg(kernel,0,sizeof(cl_mem),&input);
          clSetKernelArg(kernel,1,sizeof(cl_int),&strides[0]);
          clSetKernelArg(kernel,2,sizeof(cl_int),&strides[1]);
          clSetKernelArg(kernel,3,sizeof(cl_int),&ioffset);
          clSetKernelArg(kernel,4,sizeof(cl_mem),&output);
          clSetKernelArg(kernel,5,sizeof(cl_int),&ooffset);
          size_t gsize[2];
          gsize[0] = dims[0];
          gsize[1] = dims[1];
          clEnqueueNDRangeKernel(q,kernel,2,NULL,gsize,NULL,0,NULL,&evt);
    }else{
        cl_kernel kernel = kernels[nversions*nops*dtype + nversions*op + 2];
        cl_mem strideBuf,dimsBuf;
        strideBuf = clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR,sizeof(cl_int)*ndims,strides,NULL);
        dimsBuf = clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR,sizeof(cl_int)*ndims,dims,NULL);
        clSetKernelArg(kernel,0,sizeof(cl_mem),&input);
        clSetKernelArg(kernel,1,sizeof(cl_int),&ioffset);
        clSetKernelArg(kernel,2,sizeof(cl_mem),&output);
        clSetKernelArg(kernel,3,sizeof(cl_int),&ooffset);
        clSetKernelArg(kernel,4,sizeof(cl_int),&ndims);
        clSetKernelArg(kernel,5,sizeof(cl_mem),&strideBuf);
        clSetKernelArg(kernel,6,sizeof(cl_mem),&dimsBuf);
        size_t gsize = 1;
        for(int i=0;i<ndims;i++) gsize *= dims[i];
        clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,NULL,0,NULL,&evt);
        RaijinTempBuf *d1 = new RaijinTempBuf(strideBuf);
        RaijinTempBuf *d2 = new RaijinTempBuf(dimsBuf);
        clSetEventCallback(evt,CL_COMPLETE,RaijinTempBuf::deleteBufferCallBack,d1);
        clSetEventCallback(evt,CL_COMPLETE,RaijinTempBuf::deleteBufferCallBack,d2);
    }
    return evt;
}

static cl_kernel getKernel(string source,string kname,cl_device_id dvc,cl_context ctx){
    const char* kernelChar = source.c_str();
    size_t kernelLen = source.length();
    cl_program prg = clCreateProgramWithSource(ctx,1,&kernelChar,&kernelLen,NULL);
    clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel kernel = clCreateKernel(prg,kname.c_str(),NULL);
    return kernel;
}

RaijinUnitaryOp::RaijinUnitaryOp(cl_device_id device, cl_context context, cl_command_queue queue):dvc(device),ctx(context),q(queue){
    //for each datatype
    for(int i=0;i<3;i++){
        //for each operation
        for(int j=SQRT;j<EXP;j++){
            RaijinDtypes::DType dtype = (RaijinDtypes::DType)i;
            RaijinUnitaryOp::Op op = (RaijinUnitaryOp::Op)j;
            string dtypeName = RaijinDtypes::getDataTypeName(dtype);
            string opName = RaijinUnitaryOp::getOpName(op);

            //make one-dimensional version
            string oneDimKernelSource = makeOneDimOp(dtype,op);
            string oneDimKname = opName + dtypeName + "OneDim";
            cl_kernel oneDimKernel = getKernel(oneDimKernelSource,oneDimKname,dvc,ctx);

            //make two-dimensional version
            string twoDimKernelSource = makeTwoDimOp(dtype,op);
            string twoDimKname = opName + dtypeName + "TwoDim";
            cl_kernel twoDimKernel = getKernel(twoDimKernelSource,twoDimKname,dvc,ctx);

            //make multi-dimensional version
            string multiDimKernelSource = makeMultiDimOp(dtype,op);
            string multiDimKname = opName + dtypeName + "MultiDim";
            cl_kernel multiDimKernel = getKernel(multiDimKernelSource,multiDimKname,dvc,ctx);

            kernels.push_back(oneDimKernel);
            kernels.push_back(twoDimKernel);
            kernels.push_back(multiDimKernel);
        }
    }
}

RaijinUnitaryOp::~RaijinUnitaryOp(){
    for(size_t i=0;i<kernels.size();i++){
        cl_kernel kernel = kernels[i];
        cl_program prg;
        clGetKernelInfo(kernel,CL_KERNEL_PROGRAM,sizeof(cl_program),&prg,NULL);
        clReleaseKernel(kernel);
        clReleaseProgram(prg);
    }
}
