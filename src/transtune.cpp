#include "raijin.hpp"
#include <sstream>
#include <fstream>
using namespace std;
using namespace RaijinCL;



template <typename T,bool isDouble>
double testTransReal(cl_device_id dvc,cl_context ctx,string kernel,int simdw,bool useImg,int lx,int ly,int N=2048){
	size_t sizebuf = sizeof(T)*N*N;
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,NULL);
	size_t klen = kernel.length();
	const char *ksrc = kernel.c_str();
	cl_program prg = clCreateProgramWithSource(ctx,1,&ksrc,&klen,NULL);
	clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"transKernel",NULL);
	T *A = new T[N*N];
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			A[i*N+j] = 0;
		}
	}
	cl_mem bufIn,bufOut;
	bufIn = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizebuf,NULL,NULL);
	if(useImg){
		unsigned int ncomp =simdw;
		if(isDouble) ncomp*=2;
		cl_image_format format;
        switch(ncomp){
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
		bufOut = clCreateImage2D(ctx,CL_MEM_READ_WRITE,&format,N/simdw,N,0,NULL,NULL);
	}else{
		bufOut = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizebuf,NULL,NULL);
	}
	clEnqueueWriteBuffer(q,bufIn,CL_TRUE,0,sizebuf,A,0,NULL,NULL);
	clFinish(q);
	cl_int zero = 0;
	cl_int cln = N;
	cl_int clnbysimd = N/simdw;
	clSetKernelArg(krnl,0,sizeof(cl_mem),&bufIn);
	clSetKernelArg(krnl,1,sizeof(cl_mem),&bufOut);
	clSetKernelArg(krnl,2,sizeof(cl_int),&zero);
	clSetKernelArg(krnl,3,sizeof(cl_int),&cln);
	clSetKernelArg(krnl,4,sizeof(cl_int),&zero);
	clSetKernelArg(krnl,5,sizeof(cl_int),&clnbysimd);
	clSetKernelArg(krnl,6,sizeof(cl_int),&cln);
	clFinish(q);
	const int niters = 5;
	const size_t gsize[] = {N/simdw,N};
	const size_t lsize[] = {ly,lx};
	double tdiff = 0;
	for(int i=0;i<niters;i++){
		RTimer rt;
		rt.start();
		clEnqueueNDRangeKernel(q,krnl,2,NULL,gsize,lsize,0,NULL,NULL);
		clFinish(q);
		rt.stop();
		if(i>0) tdiff += rt.getDiff();
	}
	delete[] A;
	clReleaseMemObject(bufIn);
	clReleaseMemObject(bufOut);
	clReleaseKernel(krnl);
	clReleaseProgram(prg);
	clReleaseCommandQueue(q);
	double bw = sizebuf*(niters-1)*1.0e-9/tdiff;
	return bw;
}

template <typename T,bool isDouble>
double testTransComplex(cl_device_id dvc,cl_context ctx,string kernel,int simdw,bool useImg,int lx,int ly,int N=2048){
	size_t sizebuf = sizeof(T)*N*N;
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,NULL);
	size_t klen = kernel.length();
	const char *ksrc = kernel.c_str();
	cl_program prg = clCreateProgramWithSource(ctx,1,&ksrc,&klen,NULL);
	clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"transKernel",NULL);
	T *A = new T[N*N];
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			A[i*N+j].s[0] = 0;
			A[i*N+j].s[1] = 0;
		}
	}
	cl_mem bufIn,bufOut;
	bufIn = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizebuf,NULL,NULL);
	if(useImg){
		unsigned int ncomp =simdw*2;
		if(isDouble) ncomp*=2;
		cl_image_format format;
        switch(ncomp){
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
		bufOut = clCreateImage2D(ctx,CL_MEM_READ_WRITE,&format,N/simdw,N,0,NULL,NULL);
	}else{
		bufOut = clCreateBuffer(ctx,CL_MEM_READ_WRITE,sizebuf,NULL,NULL);
	}
	clEnqueueWriteBuffer(q,bufIn,CL_TRUE,0,sizebuf,A,0,NULL,NULL);
	clFinish(q);
	cl_int zero = 0;
	cl_int cln = N;
	cl_int clnbysimd = N/simdw;
	clSetKernelArg(krnl,0,sizeof(cl_mem),&bufIn);
	clSetKernelArg(krnl,1,sizeof(cl_mem),&bufOut);
	clSetKernelArg(krnl,2,sizeof(cl_int),&zero);
	clSetKernelArg(krnl,3,sizeof(cl_int),&cln);
	clSetKernelArg(krnl,4,sizeof(cl_int),&zero);
	clSetKernelArg(krnl,5,sizeof(cl_int),&clnbysimd);
	clSetKernelArg(krnl,6,sizeof(cl_int),&cln);
	clFinish(q);
	const int niters = 5;
	const size_t gsize[] = {N/simdw,N};
	const size_t lsize[] = {ly,lx};
	double tdiff = 0;
	for(int i=0;i<niters;i++){
		RTimer rt;
		rt.start();
		clEnqueueNDRangeKernel(q,krnl,2,NULL,gsize,lsize,0,NULL,NULL);
		clFinish(q);
		rt.stop();
		if(i>0) tdiff += rt.getDiff();
	}
	delete[] A;
	clReleaseMemObject(bufIn);
	clReleaseMemObject(bufOut);
	clReleaseKernel(krnl);
	clReleaseProgram(prg);
	clReleaseCommandQueue(q);
	double bw = sizebuf*(niters-1)*1.0e-9/tdiff;
	return bw;
}


void RaijinTranspose::tuneStrans(cl_device_id dvc){
	cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
	cl_context_properties conprop[3];
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;
    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    string dpath = raijinGetProfileFileName(dvc,"strans");
    ofstream ofile(dpath.c_str());
    //write information about the optimal transpose routine
	bool toImg[] = {true,false};
	int simdw[] = {1,2,4,8};
	bool useLocalMem[] = {true,false};
	int lx[] = {4,8,4,16,16};
	int ly[] = {4,8,16,4,16};


    //for each output type
	for(unsigned int toImgIdx=0;toImgIdx<2;toImgIdx++){
		 //for each simdwidth
		for(unsigned int simdIdx=0;simdIdx<4;simdIdx++){
			double best = 0;
            bool inited = false;
            RaijinTransOpt opt;
			for(unsigned int localMemIdx=0;localMemIdx<2;localMemIdx++){

				for(unsigned int lidx=0;lidx<5;lidx++){
					string kernel;
                    bool ret = createRealTrans(simdw[simdIdx],lx[lidx],ly[lidx],useLocalMem[localMemIdx],toImg[toImgIdx],false,kernel,isAmd64(dvc));
					if(ret){
						//cout<<"Kernel is "<<kernel<<endl;
						double bw = testTransReal<cl_float,false>(dvc,ctx,kernel,simdw[simdIdx],toImg[toImgIdx],lx[lidx],ly[lidx],2048);
						cout<<"toImg? "<<toImg[toImgIdx]<<" SIMD "<<simdw[simdIdx]<<" localMem? "<<useLocalMem[localMemIdx]<<" Group size "<<lx[lidx]<<","<<ly[lidx]<<endl;
						cout<<"Bw is "<<bw<<endl;
						if(bw>best){
							best = bw;
                            inited = true;
                            opt.kernel = kernel;
                            opt.lx = lx[lidx];
                            opt.ly = ly[lidx];
						}
					}

				}
			}
            if(inited){
                //cout<<"Best kernel for "<<simdw[simdIdx]<<" toImg "<<toImg[toImgIdx]<<" "<<best<<endl;
                //cout<<opt.kernel<<endl;
                ofile<<"start_case"<<endl;
                ofile<<toImg[toImgIdx]<<" "<<simdw[simdIdx]<<endl;
                ofile<<opt;
                ofile<<"end_case"<<endl;
            }

		}
	}
    RaijinTranspose t(dvc,ctx);
    clReleaseContext(ctx);
}

void RaijinTranspose::tuneDtrans(cl_device_id dvc){
		cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
	cl_context_properties conprop[3];
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;
    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    string dpath = raijinGetProfileFileName(dvc,"dtrans");
    ofstream ofile(dpath.c_str());
    //write information about the optimal transpose routine
	bool toImg[] = {true,false};
	int simdw[] = {1,2,4,8};
	bool useLocalMem[] = {true,false};
	int lx[] = {4,8,4,16,16};
	int ly[] = {4,8,16,4,16};


    //for each output type
	for(unsigned int toImgIdx=0;toImgIdx<2;toImgIdx++){
		 //for each simdwidth
		for(unsigned int simdIdx=0;simdIdx<3;simdIdx++){
			double best = 0;
            RaijinTransOpt opt;
            bool inited = false;

			for(unsigned int localMemIdx=0;localMemIdx<2;localMemIdx++){

				for(unsigned int lidx=0;lidx<5;lidx++){
					string kernel;
					bool ret = createRealTrans(simdw[simdIdx],lx[lidx],ly[lidx],useLocalMem[localMemIdx],toImg[toImgIdx],true,kernel,isAmd64(dvc));
					if(ret){
						//cout<<"Kernel is "<<kernel<<endl;
						double bw = testTransReal<cl_double,true>(dvc,ctx,kernel,simdw[simdIdx],toImg[toImgIdx],lx[lidx],ly[lidx],2048);
						cout<<"toImg? "<<toImg[toImgIdx]<<" SIMD "<<simdw[simdIdx]<<" localMem? "<<useLocalMem[localMemIdx]<<" Group size "<<lx[lidx]<<","<<ly[lidx]<<endl;
						cout<<"Bw is "<<bw<<endl;
						if(bw>best){
                            opt.kernel = kernel;
                            opt.lx = lx[lidx];
                            opt.ly = ly[lidx];
							best = bw;
                            inited = true;
						}
					}

				}
			}
            if(inited){
                ofile<<"start_case"<<endl;
                ofile<<toImg[toImgIdx]<<" "<<simdw[simdIdx]<<endl;
                ofile<<opt;
                ofile<<"end_case"<<endl;
            }

		}
    }
    RaijinTranspose t(dvc,ctx);
    clReleaseContext(ctx);

}

void RaijinTranspose::tuneCtrans(cl_device_id dvc){
	cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
	cl_context_properties conprop[3];
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;
    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    string dpath = raijinGetProfileFileName(dvc,"ctrans");
    ofstream ofile(dpath.c_str());
    //write information about the optimal transpose routine
	bool toImg[] = {true,false};
	int simdw[] = {1,2,4,8};
	bool useLocalMem[] = {true,false};
	int lx[] = {4,8,4,16,16};
	int ly[] = {4,8,16,4,16};


    //for each output type
	for(unsigned int toImgIdx=0;toImgIdx<2;toImgIdx++){
		 //for each simdwidth
		for(unsigned int simdIdx=0;simdIdx<3;simdIdx++){
			double best = 0;
            RaijinTransOpt opt;
            bool inited = false;

			for(unsigned int localMemIdx=0;localMemIdx<2;localMemIdx++){

				for(unsigned int lidx=0;lidx<5;lidx++){
					string kernel;
					bool ret = createComplexTrans(simdw[simdIdx],lx[lidx],ly[lidx],useLocalMem[localMemIdx],toImg[toImgIdx],false,kernel);
					if(ret){
						//cout<<"Kernel is "<<kernel<<endl;
						double bw = testTransComplex<cl_float2,false>(dvc,ctx,kernel,simdw[simdIdx],toImg[toImgIdx],lx[lidx],ly[lidx],2048);
						cout<<"toImg? "<<toImg[toImgIdx]<<" SIMD "<<simdw[simdIdx]<<" localMem? "<<useLocalMem[localMemIdx]<<" Group size "<<lx[lidx]<<","<<ly[lidx]<<endl;
						cout<<"Bw is "<<bw<<endl;
						if(bw>best){
                            opt.kernel = kernel;
                            opt.lx = lx[lidx];
                            opt.ly = ly[lidx];
							best = bw;
                            inited = true;
						}
					}

				}
			}
            if(inited){
                ofile<<"start_case"<<endl;
                ofile<<toImg[toImgIdx]<<" "<<simdw[simdIdx]<<endl;
                ofile<<opt;
                ofile<<"end_case"<<endl;
            }

		}
    }
    RaijinTranspose t(dvc,ctx);
    clReleaseContext(ctx);


}

void RaijinTranspose::tuneZtrans(cl_device_id dvc){
	cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
	cl_context_properties conprop[3];
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;
    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    string dpath = raijinGetProfileFileName(dvc,"ztrans");
    ofstream ofile(dpath.c_str());
    //write information about the optimal transpose routine
	bool toImg[] = {true,false};
	int simdw[] = {1,2,4,8};
	bool useLocalMem[] = {true,false};
	int lx[] = {4,8,4,16,16};
	int ly[] = {4,8,16,4,16};


    //for each output type
	for(unsigned int toImgIdx=0;toImgIdx<2;toImgIdx++){
		 //for each simdwidth
		for(unsigned int simdIdx=0;simdIdx<3;simdIdx++){
			double best = 0;
            RaijinTransOpt opt;
            bool inited = false;

			for(unsigned int localMemIdx=0;localMemIdx<2;localMemIdx++){

				for(unsigned int lidx=0;lidx<5;lidx++){
					string kernel;
					bool ret = createComplexTrans(simdw[simdIdx],lx[lidx],ly[lidx],useLocalMem[localMemIdx],toImg[toImgIdx],true,kernel);
					if(ret){
						//cout<<"Kernel is "<<kernel<<endl;
						double bw = testTransComplex<cl_double2,true>(dvc,ctx,kernel,simdw[simdIdx],toImg[toImgIdx],lx[lidx],ly[lidx],2048);
						cout<<"toImg? "<<toImg[toImgIdx]<<" SIMD "<<simdw[simdIdx]<<" localMem? "<<useLocalMem[localMemIdx]<<" Group size "<<lx[lidx]<<","<<ly[lidx]<<endl;
						cout<<"Bw is "<<bw<<endl;
						if(bw>best){
                            opt.kernel = kernel;
                            opt.lx = lx[lidx];
                            opt.ly = ly[lidx];
							best = bw;
                            inited = true;
						}
					}

				}
			}
            if(inited){
                ofile<<"start_case"<<endl;
                ofile<<toImg[toImgIdx]<<" "<<simdw[simdIdx]<<endl;
                ofile<<opt;
                ofile<<"end_case"<<endl;
            }

		}
    }
    RaijinTranspose t(dvc,ctx);
    clReleaseContext(ctx);


}
