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

#include <string>
#include <sstream>
#include <fstream>
#include "raijin.hpp"

using namespace std;
using namespace RaijinCL;
void RaijinTempBuf::deleteBufferCallBack(cl_event evt, cl_int status, void *vdata){
    RaijinTempBuf *tempBuf = (RaijinTempBuf*)vdata;
    delete tempBuf;
}

RaijinTempBuf::~RaijinTempBuf(){
    clReleaseMemObject(buf);
}


string RaijinReduce::getOpName(Op op){
    switch(op){
    case SUM:
        return "Sum";
    case PROD:
        return "Prod";
    }
}

string RaijinReduce::getOpSymbol(Op op){
    switch(op){
    case SUM:
        return "+";
    case PROD:
        return "*";
    }
}

string RaijinDtypes::getDataTypeName(RaijinDtypes::DType dtype){
    switch(dtype){
    case DOUBLE:
        return "Double";
    case FLOAT:
        return "Float";
    case COMPLEX:
        return "Complex";
    }
}

string RaijinDtypes::getDataTypeSym(RaijinDtypes::DType dtype){
    switch(dtype){
    case DOUBLE:
        return "double";
    case FLOAT:
        return "float";
    case COMPLEX:
        return "double2";
    }
}

static string makeReduceOneDimReal(RaijinDtypes::DType datatype,RaijinReduce::Op optype){
    stringstream ss;
    const string dtype = RaijinDtypes::getDataTypeSym(datatype);
    const string dtypeName = RaijinDtypes::getDataTypeName(datatype);
    const string op = RaijinReduce::getOpSymbol(optype);
    const string opname = RaijinReduce::getOpName(optype);

    if(datatype==RaijinDtypes::DOUBLE){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    int init = 0;
    if(optype==RaijinReduce::PROD){
        init = 1;
    }
    ss<<"__kernel void "<<opname<<"OneDim"<<dtypeName<<"(__global "<<dtype<<" *input, int ioffset, __global "<<dtype<<" *output, int ooffset, int stride, int start, int lenModStripLen){"<<endl;
    ss<<"const int stripLen = 16;"<<endl;
    ss<<"int id = get_global_id(0);"<<endl;
    ss<<"int i;"<<endl;
    ss<<dtype<<" sum = "<<init<<";"<<endl;
    ss<<"if(id==0){"<<endl;
    ss<<"for(i=0;i<lenModStripLen;i++) sum "<<op<<"= input[ioffset+stride*i];"<<endl;
    ss<<"}else{"<<endl;
    ss<<"for(i=0;i<stripLen;i+=4){"<<endl;
    ss<<"int start = ioffset + stride*(lenModStripLen + (id-1)*stripLen);"<<endl;
    ss<<" sum "<<op<<"= input[start+stride*(i+0)];"<<endl;
    ss<<" sum "<<op<<"= input[start+stride*(i+1)];"<<endl;
    ss<<" sum "<<op<<"= input[start+stride*(i+2)];"<<endl;
    ss<<" sum "<<op<<"= input[start+stride*(i+3)];"<<endl;
    ss<<"}"<<endl;
    ss<<"}"<<endl;
    ss<<"output[ooffset+ id] = sum;"<<endl;
    ss<<" }"<<endl;
    return ss.str();
}

/*For 2 dimensions, each element of the result vector is computed by a work-group. Each work-group performs a two-phase reduction.
  In the first phase of the two-phase reduction, the threads in the work-group each compute a part of the final element.
  The threads write it to local memory. Then in the second phase, the 0th thread of the work-group sums the partial sums.

  For 3 dimensions, the problem is actually very similar. Each work-group will compute one element of the final result matrix is computed
  by a work-group. Just that the computation of the address of the element is more complex.
  */

/*
__kernel
void sumTwoDim(__global float *input, int ioffset, __global float *output, int ooffset, int stride1,int stride2, int len,int lenDivGroupSize,int lenModGroupSize){
    size_t idx = get_group_id(0);
    size_t lidx = get_local_id(0);
    int start = idx*stride1 + lidx*lenDivGroupSize*stride2+ioffset;
    const int wgsize = 16;
    __local float temp[wgsize];
    int i;
    float threadSum = 0;
    for(i=0;i<lenDivGroupSize;i++){
        threadSum += input[start + i*stride2];
    }
    temp[lidx] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx==0){
        float finalSum = 0;
        for(i=0;i<wgsize;i++) finalSum += temp[i];
        start = start + wgsize*lenDivGroupSize*stride2;
        for(i=0;i<lenModGroupSize;i++) finalSum += input[start + i*stride2];
        output[ooffset + idx] = finalSum;
    }
}*/

static string makeReduceTwoDimReal(RaijinDtypes::DType datatype,RaijinReduce::Op optype){
    stringstream ss;

    const string dtype = RaijinDtypes::getDataTypeSym(datatype);
    const string dtypeName = RaijinDtypes::getDataTypeName(datatype);
    const string op = RaijinReduce::getOpSymbol(optype);
    const string opname = RaijinReduce::getOpName(optype);
    if(datatype==RaijinDtypes::DOUBLE){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    int init = 0;
    if(optype==RaijinReduce::PROD){
        init = 1;
    }
    ss<<"__kernel void "<<opname<<"TwoDim"<<dtypeName<<"(__global "<<dtype<<" *input, int ioffset, __global "<<dtype<<" *output, int ooffset,int stride1,int stride2, int len,int lenDivGroupSize,int lenModGroupSize){"<<endl;
    ss<<" size_t idx = get_group_id(0);"<<endl;
    ss<<" size_t lidx = get_local_id(0);"<<endl;
    ss<<" int start = idx*stride1 + lidx*lenDivGroupSize*stride2+ioffset;"<<endl;
    ss<<" const int wgsize = 16;"<<endl;
    ss<<" __local "<<dtype<<" temp[wgsize];"<<endl;
    ss<<" int i;"<<endl;
    ss<<" "<<dtype<<" threadSum = "<<init<<";"<<endl;
    ss<<" for(i=0;i<lenDivGroupSize;i++){"<<endl;
    ss<<"  threadSum "<<op<<"= input[start + i*stride2];"<<endl;
    ss<<" }"<<endl;
    ss<<" temp[lidx] = threadSum;"<<endl;
    ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<" if(lidx==0){"<<endl;
    ss<<"   "<<dtype<<" finalSum = 0;"<<endl;
    ss<<"   for(i=0;i<wgsize;i++) finalSum "<<op<<"= temp[i];"<<endl;
    ss<<"   start = start + wgsize*lenDivGroupSize*stride2;"<<endl;
    ss<<"   for(i=0;i<lenModGroupSize;i++) finalSum "<<op<<"= input[start + i*stride2];"<<endl;
    ss<<"   output[ooffset + idx] = finalSum;"<<endl;
    ss<<" }"<<endl;
    ss<<"}"<<endl;
    return ss.str();
}


/*
__kernel
void sumMultiDimFloat(__global float *input, int ioffset, __global float *output, int ooffset, int ndims, int axis, __global const int *strides, __global const int *dims,
int len, int lenDivGroupSize, int lenModGroupSize){
    size_t groupidx = get_group_id(0);
    size_t lidx = get_local_id(0);
    int oidx = ooffset;
    int start = ioffset;
    const int wgsize = 16;
    __local float temp[wgsize];
    int i;
    int curidx = groupidx;
    int multiplier = 1;
    for(i=ndims-1;i>=0;i--){
        if(i==axis) continue;
        int dimsize = dims[i];
        int dimidx = curidx%dimsize;
        curidx = curidx/dimsize;
        start += dimidx*strides[i];
        oidx += dimidx*multiplier;
        multiplier *= dims[i];
    }
    int axisstride = strides[axis];
    float threadSum = 0;
    for(i=0;i<lenDivGroupSize;i++){
        threadSum += input[start + i*axisstride];
    }
    temp[lidx] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx==0){
        float finalSum = 0;
        for(i=0;i<wgsize;i++) finalSum += temp[i];
        start = start + wgsize*lenDivGroupSize*axisstride;
        for(i=0;i<lenModGroupSize;i++) finalSum += input[start + i*axisstride];
        output[ooffset + oidx] = finalSum;
    }
}*/

static string makeReduceMultiDimReal(RaijinDtypes::DType datatype,RaijinReduce::Op optype){
    stringstream ss;
    const string dtype = RaijinDtypes::getDataTypeSym(datatype);
    const string dtypeName = RaijinDtypes::getDataTypeName(datatype);
    const string op = RaijinReduce::getOpSymbol(optype);
    const string opname = RaijinReduce::getOpName(optype);

    if(datatype==RaijinDtypes::DOUBLE){
        ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
    }
    int init = 0;
    if(optype==RaijinReduce::PROD){
        init = 1;
    }
    ss<<"__kernel ";
    ss<<"void sumMultiDimFloat(__global float *input, int ioffset, __global float *output, int ooffset, int ndims, int axis, __global const int *strides, __global const int *dims,";
    ss<<" int len, int lenDivGroupSize, int lenModGroupSize){"<<endl;
    ss<<" size_t groupidx = get_group_id(0);"<<endl;
    ss<<" size_t lidx = get_local_id(0);"<<endl;
    ss<<" int oidx = ooffset;"<<endl;
    ss<<" int start = ioffset;"<<endl;
    ss<<" const int wgsize = 16;"<<endl;
    ss<<" __local float temp[wgsize];"<<endl;
    ss<<" int i;"<<endl;
    ss<<" int curidx = groupidx;"<<endl;
    ss<<" int multiplier = 1;"<<endl;
    ss<<" for(i=ndims-1;i>=0;i--){"<<endl;
    ss<<"   if(i==axis) continue;"<<endl;
    ss<<"   int dimsize = dims[i];"<<endl;
    ss<<"   int dimidx = curidx%dimsize;"<<endl;
    ss<<"   curidx = curidx/dimsize;"<<endl;
    ss<<"   start += dimidx*strides[i];"<<endl;
    ss<<"   oidx += dimidx*multiplier;"<<endl;
    ss<<"   multiplier *= dims[i];"<<endl;
    ss<<" }"<<endl;
    ss<<" int axisstride = strides[axis];"<<endl;
    ss<<" float threadSum = "<<init<<";"<<endl;
    ss<<" for(i=0;i<lenDivGroupSize;i++){"<<endl;
    ss<<"   threadSum += input[start + i*axisstride];"<<endl;
    ss<<" }";
    ss<<" temp[lidx] = threadSum;"<<endl;
    ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<" if(lidx==0){"<<endl;
    ss<<"   float finalSum = 0;"<<endl;
    ss<<"   for(i=0;i<wgsize;i++) finalSum += temp[i];"<<endl;
    ss<<"   start = start + wgsize*lenDivGroupSize*axisstride;"<<endl;
    ss<<"   for(i=0;i<lenModGroupSize;i++) finalSum += input[start + i*axisstride];"<<endl;
    ss<<"   output[ooffset + oidx] = finalSum;"<<endl;
    ss<<" }"<<endl;
    ss<<"}"<<endl;
    return ss.str();
}


cl_event RaijinReduce::apply(Op op, RaijinDtypes::DType dtype, cl_mem input, cl_int ioffset, cl_mem output, cl_int ooffset, cl_mem buf, cl_int ndims, cl_int *dims, cl_int *strides, cl_int axis){
    if(ndims==1){
        cl_kernel kernel = kernels[6*dtype + 3*op];
        const int stripLen = 16;
        int curLen = dims[0];
        int bufOffset = 0;

        cl_mem curInput = input;
        cl_int curInOffset = ioffset;
        cl_int curOutOffset = bufOffset;
        bool hasPrevEvent = false;
        cl_event prevEvent;
        cl_int lenModGroupSize;
        cl_event evt;
        size_t gsize;
        const size_t lsize = 16;
        while(curLen>stripLen){
            //enqueue kernel
            clSetKernelArg(kernel,0,sizeof(cl_mem),&curInput);
            clSetKernelArg(kernel,1,sizeof(cl_int),&curInOffset);
            clSetKernelArg(kernel,2,sizeof(cl_mem),&buf);
            clSetKernelArg(kernel,3,sizeof(cl_int),&curOutOffset);
            lenModGroupSize = curLen%16;
            clSetKernelArg(kernel,4,sizeof(cl_int),&lenModGroupSize);
            gsize = curLen;
            if(hasPrevEvent){
                clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,1,&prevEvent,&evt);
            }else{
                clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,0,NULL,&evt);
            }
            prevEvent = evt;
            hasPrevEvent = true;
            //new curLen is the ceil of curLen/stripLen
            curLen = curLen/stripLen + (curLen%stripLen>0 ? 1:0);
            curInput = buf;
            curInOffset = bufOffset;
            bufOffset = bufOffset + curLen;
            curOutOffset = bufOffset;
        }
        //enqueue summation into output buffer
        clSetKernelArg(kernel,0,sizeof(cl_mem),&curInput);
        clSetKernelArg(kernel,1,sizeof(cl_int),&curInOffset);
        clSetKernelArg(kernel,2,sizeof(cl_mem),&output);
        clSetKernelArg(kernel,3,sizeof(cl_int),&ooffset);
        lenModGroupSize = curLen%16;
        clSetKernelArg(kernel,4,sizeof(cl_int),&lenModGroupSize);
        gsize = curLen;
        if(hasPrevEvent){
            clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,1,&prevEvent,&evt);
        }else{
            clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,0,NULL,&evt);
        }
        return evt;
    }else if(ndims==2){
        cl_kernel kernel = kernels[6*dtype + 3*op+1];
        clSetKernelArg(kernel,0,sizeof(cl_mem),&input);
        clSetKernelArg(kernel,1,sizeof(cl_int),&ioffset);
        clSetKernelArg(kernel,2,sizeof(cl_mem),&output);
        clSetKernelArg(kernel,3,sizeof(cl_int),&ooffset);
        cl_int stride1,stride2,len,lenDivGroupSize,lenModGroupSize;
        size_t gsize;
        if(axis==0){
            stride1 = strides[1];
            stride2 = strides[0];
            len = dims[0];
            gsize = dims[1];
        }else if(axis==1){
            stride1 = strides[0];
            stride2 = strides[1];
            len = dims[1];
            gsize = dims[0];
        }
        lenModGroupSize = len%16;
        lenDivGroupSize = len/16;
        clSetKernelArg(kernel,4,sizeof(cl_int),&stride1);
        clSetKernelArg(kernel,5,sizeof(cl_int),&stride2);
        clSetKernelArg(kernel,6,sizeof(cl_int),&len);
        clSetKernelArg(kernel,7,sizeof(cl_int),&lenDivGroupSize);
        clSetKernelArg(kernel,8,sizeof(cl_int),&lenModGroupSize);
        cl_event evt;
        size_t lsize = 16;
        clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,0,NULL,&evt);
        return evt;

    }else{
        cl_kernel kernel = kernels[6*dtype + 3*op+2];
        clSetKernelArg(kernel,0,sizeof(cl_mem),&input);
        clSetKernelArg(kernel,1,sizeof(cl_int),&ioffset);
        clSetKernelArg(kernel,2,sizeof(cl_mem),&output);
        clSetKernelArg(kernel,3,sizeof(cl_int),&ooffset);
        clSetKernelArg(kernel,4,sizeof(cl_int),&ndims);
        clSetKernelArg(kernel,5,sizeof(cl_int),&axis);
        cl_mem strideBuf,dimsBuf;
        strideBuf = clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR,sizeof(cl_int)*ndims,(void *)strides,NULL);
        dimsBuf = clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR,sizeof(cl_int)*ndims,(void *)dims,NULL);
        clSetKernelArg(kernel,6,sizeof(cl_mem),&strideBuf);
        clSetKernelArg(kernel,7,sizeof(cl_mem),&dimsBuf);
        cl_int len = dims[axis];
        cl_int lenModGroupSize  = len%16;
        cl_int lenDivGroupSize = len/16;
        clSetKernelArg(kernel,8,sizeof(cl_int),&len);
        clSetKernelArg(kernel,9,sizeof(cl_int),&lenDivGroupSize);
        clSetKernelArg(kernel,10,sizeof(cl_int),&lenModGroupSize);
        size_t lsize = 16;
        size_t gsize = 1;
        for(int i=0;i<ndims;i++){
            if(i==axis) continue;
            gsize *= dims[i];
        }
        cl_event evt;
        clEnqueueNDRangeKernel(q,kernel,1,NULL,&gsize,&lsize,0,NULL,&evt);
        RaijinTempBuf *d1 = new RaijinTempBuf(strideBuf);
        RaijinTempBuf *d2 = new RaijinTempBuf(dimsBuf);
        clSetEventCallback(evt,CL_COMPLETE,RaijinTempBuf::deleteBufferCallBack,d1);
        clSetEventCallback(evt,CL_COMPLETE,RaijinTempBuf::deleteBufferCallBack,d2);
        return evt;
    }
}

static string readKernelFromFile(const string& fname){
    ifstream file(fname.c_str());
    string line;
    stringstream kstream;
    while(file.good()){
        getline(file,line);
        kstream<<line<<" "<<endl;
    }
    file.close();
    return kstream.str();
}

RaijinReduce::RaijinReduce(cl_device_id device, cl_context context, cl_command_queue queue): dvc(device),ctx(context),q(queue),kernels(18){
    clRetainContext(ctx);
    clRetainCommandQueue(q);

    string dtypeNames[3];
    dtypeNames[0] = "Float";
    dtypeNames[1] = "Double";
    dtypeNames[2] = "Complex";

    string opNames[2];
    opNames[0] = "sum";
    opNames[1] = "prod";

    string variantNames[3];
    variantNames[0] = "OneDim";
    variantNames[1] = "TwoDim";
    variantNames[2] = "MultiDim";
    const string dname = getenv("RAIJIN_TUNE_PATH");

    //for each datatype
    for(int i=0;i<3;i++){
        //for each operation
        for(int j=0;j<2;j++){
            //for each variant: one-dim, two-dim, multi-dim
            RaijinDtypes::DType dtype = (RaijinDtypes::DType)i;
            RaijinReduce::Op optype = (RaijinReduce::Op)j;
            for(int k=0;k<3;k++){
                string kname = opNames[j] + variantNames[k] + dtypeNames[i];
                string kernelSource;
                if(dtype==RaijinDtypes::COMPLEX){
                    string fname = dname + kname + ".cl";
                    kernelSource = readKernelFromFile(fname);
                }else{
                    switch(j){
                    case 0:
                        kernelSource = makeReduceOneDimReal(dtype,optype);
                        break;
                    case 1:
                        kernelSource = makeReduceTwoDimReal(dtype,optype);
                        break;
                    case 2:
                        kernelSource = makeReduceMultiDimReal(dtype,optype);
                        break;
                    }
                }
                const char *kernelSrcCharPtr = kernelSource.c_str();
                size_t kernelLen = kernelSource.length();
                cl_program prg = clCreateProgramWithSource(ctx,1,&kernelSrcCharPtr,&kernelLen,NULL);
                clBuildProgram(prg,1,&dvc,"",NULL,NULL);
                cl_kernel kernel = clCreateKernel(prg,kname.c_str(),NULL);
                kernels[6*i + 3*j + k] = kernel;
            }
        }
    }
}

RaijinReduce::~RaijinReduce(){
    for(int i=0;i<kernels.size();i++){
        cl_kernel kernel = kernels[i];
        cl_program prg;
        clGetKernelInfo(kernel,CL_KERNEL_PROGRAM,sizeof(cl_program),&prg,NULL);
        clReleaseKernel(kernel);
        clReleaseProgram(prg);
    }
    clReleaseContext(ctx);
    clReleaseCommandQueue(q);
}

