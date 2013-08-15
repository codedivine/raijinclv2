
#ifdef RAIJIN_EXPERIMENTAL
extern "C"{
#include <acml.h>
}
#pragma comment(lib,"libacml_mp_dll")
#endif

#include "raijin.hpp"
#include "raijin_complex.hpp"
#include <iostream>
#include <cmath>
#include "rtimer.hpp"


#include <malloc.h>

using namespace std;
using namespace RaijinCL;

int cpuPart = 0;
int gpuPart = 0;


template <typename T>
void pattern1(T *A, T *B,int M,int K,int N){
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			A[i*K+j] = (i*1.0)/K;
		}
	}
	for(int i=0;i<K;i++){
		for(int j=0;j<N;j++){
			B[i*N+j] = (j*1.0)/K;
		}
	}
}

template <typename T>
double verifyPattern1(T *A,T *B, T *C,int M,int N,int K){
	double sum = 0.0;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			double expected = K*(i*1.0/K)*(j*1.0/K);
			double read = C[i*N+j];
			//if(i==1) cout<<"Read "<<read<<endl;
            double diff = abs(read-expected);
            if(diff>5.0) {
                //cout<<i<<" "<<j<<" "<<expected<<" "<<read<<" "<<endl;
                //exit(1);
            }
            if(diff>sum) sum = diff;
		}
	}
    return sum;
}

template <typename T>
void pattern2(T *A, T *B,int M,int K,int N){
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			A[i*K+j] = (i+j)*1.0/K;
		}
	}
	for(int i=0;i<K;i++){
		for(int j=0;j<N;j++){
			B[i*N+j] = (i+j)*1.0/K;
		}
	}
}

template <typename T>
double verifyPattern2(T *A,T *B, T *C,int M,int N,int K){
	cout<<"Verifying"<<endl;
	double sum = 0.0;
	double m = 1.0/K;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			double expected = 0.0;
			for(int k=0;k<K;k++){
				expected += ((i+k)*m)*((k+j)*m);
			}
			double read = C[i*N+j];
			double error = abs(read-expected);
            //if(error>1.0) cout<<i<<" "<<j<<" "<<read<<" "<<expected<<endl;
            double diff = abs(read-expected);
            if(diff>sum) sum = diff;
		}
	}
    return sum;
}

template <typename R>
void transpose(R *mat,int M){
    for(int i=0;i<M;i++){
        for(int j=0;j<i;j++){
            R temp = mat[i*M+j];
            mat[i*M+j] = mat[j*M+i];
            mat[j*M+i] = temp;
        }
    }
}

struct TestGemmZ{
    typedef cl_double2 ctype;
    typedef cl_double basetype;
    typedef RaijinZgemm GemmType;

};

struct TestGemmC{
    typedef cl_float2 ctype;
    typedef cl_float basetype;
    typedef RaijinCgemm GemmType;

};

template <typename T>
static void testGemmComplex(cl_device_id dvc,unsigned int N,bool transA,bool transB,bool verify=true){
    typedef typename T::ctype ctype;
    typedef typename T::basetype basetype;
    size_t size = sizeof(ctype) * N * N;
    cl_mem bufA, bufB, bufC;
    ctype *ptrA = new ctype[N * N];
    ctype *ptrB = new ctype[N * N];
    ctype *ptrC = new ctype[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(transA){
                ptrA[i * N + j].s[0] = 0.002 * j;
                ptrA[i*N +j].s[1] = 1;
            }else{
                ptrA[i*N + j].s[0] = 0.002*i;
                ptrA[i*N+j].s[1] = 1;
            }

            if(transB){
                ptrB[i * N + j].s[0] = 0.002 * i;
                ptrB[i*N+j].s[1] = 1;
            }else{
                ptrB[i*N + j].s[0] = 0.002*j;
                ptrB[i*N+j].s[1] = 1;
            }
            ptrC[i * N + j].s[0] = 0;
            ptrC[i*N+j].s[1] = 0;
        }
    }
    cl_context_properties conprop[3];
    cl_platform_id platform;
    clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL);
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;

    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,NULL);
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &errcode);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &errcode);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &errcode);
    clEnqueueWriteBuffer(q, bufA, CL_TRUE, 0, size, ptrA, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bufB, CL_TRUE, 0, size, ptrB, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bufC, CL_TRUE, 0, size, ptrC, 0, NULL, NULL);
    clFlush( q);
    typedef typename T::GemmType GemmType;
    GemmType *gemm = GemmType::getInstance(ctx,dvc);
    const int niters = 5;
    double tdiff = 0;
    for(int i=0;i<niters;i++){
        RTimer rt;
        rt.start();
        ctype alpha;
        alpha.s[0] = 1;
        alpha.s[1] = 0;
        ctype beta;
        beta.s[0] = 0;
        beta.s[1] = 0;
		RaijinCleaner *cleaner;
        cl_event evt = gemm->apply(q,&cleaner,RaijinCL::RaijinRowMajor,transA,transB,N,N,N,
                                                    alpha,bufA,N,0,bufB,N,0,beta,bufC,N,0);
        clFinish(q);
		if(cleaner!=NULL) delete cleaner;
        rt.stop();
        if(i>0) tdiff += rt.getDiff();
        cout<<"Time "<<rt.getDiff()<<endl;
    }
    tdiff /= (niters-1);
    double totalerror = 0.0;
    if(verify){
        clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, size, ptrC, 0, NULL, NULL);

        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                basetype calc = ptrC[i*N+j].s[0];
                basetype expected = N*((0.002*i)*(0.002*j)-1);
                double val = calc - expected;
                if(val<0) val = -val;
                //if(val>1) cout<<"Real: "<<i<<" "<<j<<" "<<calc<<" "<<expected<<endl;
                //if(val>1) exit(-1);
                basetype calcimg = ptrC[i*N+j].s[1];
                basetype expimg = N*(0.002*i+0.002*j);
                double valimg = calcimg - expimg;
                if(valimg<0) valimg *= -1;
                totalerror += (val+valimg);
            }
        }
    }
    double avgerror = (totalerror)/(N*N);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    delete[] ptrA;
    delete[] ptrB;
    delete[] ptrC;
    clReleaseCommandQueue(q);
    cout<<"Avg error "<<avgerror<<" Gflops "<<(8.0e-9*N*(1.0*N)*(1.0*N)/tdiff)<<" Time "<<tdiff<<endl;
}
template <typename T>
void testGemm(cl_device_id dvc,int N,int pattern,bool transA,bool transB){
	const int M = N;
	const int K = N;
	typedef typename T::realtype realtype;
	realtype *A = new realtype[M*K];
	realtype *B = new realtype[K*N];
	realtype *C = new realtype[M*N];

	switch(pattern){
		case 1:
			pattern1<realtype>(A,B,N,N,N);
			break;
		case 2:
			pattern2<realtype>(A,B,N,N,N);
			break;
	}
    if(transA){
        transpose<realtype>(A,N);
    }
    if(transB){
        transpose<realtype>(B,N);
    }

	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			C[i*N+j] = 0;
		}
	}
	cl_context_properties conprop[3];
	cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL);
	conprop[0] = CL_CONTEXT_PLATFORM;
	conprop[1] = (cl_context_properties)platform;
	conprop[2] = (cl_context_properties)0;
	cl_int errcode;

	cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,&errcode);
	cl_mem bufA, bufB, bufC;
	const int sizeA = sizeof(realtype)*M*K;
	const int sizeB = sizeof(realtype)*K*N;
	const int sizeC = sizeof(realtype)*M*N;
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeA, NULL, &errcode);
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeB, NULL, &errcode);
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeC, NULL, &errcode);
	clEnqueueWriteBuffer(q, bufA, CL_TRUE, 0, sizeA, A, 0, NULL, NULL);
	clEnqueueWriteBuffer(q, bufB, CL_TRUE, 0, sizeB, B, 0, NULL, NULL);
	clEnqueueWriteBuffer(q, bufC, CL_TRUE, 0, sizeC, C, 0, NULL, NULL);
	clFlush( q);
	typedef typename T::gemmtype Gemmtype;
	Gemmtype *gemm = Gemmtype::getInstance(ctx,dvc);
	//gemm->printOptKernelBinary();
    const int niters = 5;
    double tdiff = 0;
    for(int i=0;i<niters;i++){
        RTimer rt;
        rt.start();
		RaijinCleaner *cleaner;
        cl_event evt = gemm->apply(q,&cleaner,RaijinRowMajor,transA,transB,
                                   M,N,K,
                                   1,bufA,K,0,
                                   bufB,N,0,
                                   0,
                                   bufC,N,0);
        //clFlush(q);
        clFinish(q);
		if(cleaner!=NULL) delete cleaner;
        rt.stop();
		cout<<"Time "<<rt.getDiff()<<endl;
        if(i>1)
            tdiff += rt.getDiff();
    }
    tdiff = tdiff/(niters-2);
	clEnqueueReadBuffer(q,bufC,CL_TRUE,0,sizeC,C,0,NULL,NULL);
	delete gemm;
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);
	double error = 0.0;
	switch(pattern){
		case 1:
			error = verifyPattern1<realtype>(A,B,C,N,N,N);
			break;
		case 2:
			error = verifyPattern2<realtype>(A,B,C,N,N,N);
			break;
	}
    cout<<"Max error "<<error<<" GFlops "<<(2.0e-9*M*N*K/tdiff)<<" time "<<tdiff<<endl;
	delete[] A;
	delete[] B;
	delete[] C;
}

#ifdef RAIJIN_EXPERIMENTAL
struct BlasParams{
	//CBLAS_ORDER order;
	char transA;
	char transB;
	int M;
	int N;
	int K;
	float alpha;
	float *A;
	int lda;
	float *B;
	int ldb;
	float beta;
	float *C;
	int ldc;
};

DWORD WINAPI BlasFun(LPVOID params){
	BlasParams *myParams = (BlasParams*)params;
	sgemm(myParams->transA,myParams->transB,myParams->M,myParams->N,myParams->K,myParams->alpha,myParams->A,
		myParams->lda,myParams->B,myParams->ldb,myParams->beta,myParams->C,myParams->ldc);
	return 0;

}



void testPartSgemm(cl_device_id dvc,int N,int pattern,bool transA,bool transB){
	//mkl_set_num_threads(8);
	cl_uint refCount;
	cout<<"testPartSgemm "<<transA<<" "<<transB<<endl;
	const int M = N;
	const int K = N;
	float *A = (float*)_aligned_malloc(sizeof(float)*M*K,4096);
	float *B = (float*)_aligned_malloc(sizeof(float)*N*K,4096);
	float *C = (float*)_aligned_malloc(sizeof(float)*M*N,4096);
	size_t aptr = (size_t)A;
	size_t bptr = (size_t)B;
	size_t cptr = (size_t)C;
	cout<<"aptr "<<(aptr%4096)<<" bptr "<<(bptr%4096)<<" cptr "<<(cptr%4096)<<endl;

	switch(pattern){
		case 1:
			pattern1<float>(A,B,N,N,N);
			break;
		case 2:
			pattern2<float>(A,B,N,N,N);
			break;
	}
    if(transA){
        transpose<float>(A,N);
    }
    if(transB){
        transpose<float>(B,N);
    }

	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			C[i*N+j] = 0;
		}
	}
	cl_context_properties conprop[3];
	cl_platform_id platform;
	clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(platform),&platform,NULL);
	conprop[0] = CL_CONTEXT_PLATFORM;
	conprop[1] = (cl_context_properties)platform;
	conprop[2] = (cl_context_properties)0;
	cl_int errcode;

	cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
	clGetContextInfo(ctx,CL_CONTEXT_REFERENCE_COUNT,sizeof(cl_uint),&refCount,NULL);
	cout<<"Reference count "<<refCount<<endl;
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,&errcode);
	cl_mem bufA, bufB, bufC;
	const int sizeA = sizeof(float)*M*K;
	const int sizeB = sizeof(float)*K*N;
	const int sizeC = sizeof(float)*M*N;

	RaijinSgemm *gemm = RaijinSgemm::getInstance(ctx,dvc);

	RTimer bufTimer;
	bufTimer.start();
	RTimer rt;
    rt.start();
	cout<<"Creating buffers CPU: "<<cpuPart<<" GPU: "<<gpuPart<<endl;
	bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, A, &errcode);
	bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, B, &errcode);
	bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeC*gpuPart/(cpuPart+gpuPart), C, &errcode);
	clEnqueueWriteBuffer(q, bufC, CL_TRUE, 0, sizeC, C, 0, NULL, NULL);
	clFlush( q);
	bufTimer.stop();
	//gemm->printOptKernelBinary();
	RaijinParams params;
	params.num_events = 0;
	params.queue = q;
	params.waitEvents = NULL;
    const int niters = 1;
    double tdiff = 0;
	//void *ptr;
    for(int i=0;i<niters;i++){
        
		int lda = transA ? M : K;
		int ldb = transB ? K : N;
		//ptr = clEnqueueMapBuffer(q,bufC,CL_TRUE,0,0,sizeC,0,NULL,NULL,NULL);
		 cl_event evt = gemm->apply(RaijinRowMajor,transA,transB,
                                   M*gpuPart/(cpuPart+gpuPart),N,K,
                                   1,bufA,lda,0,
                                   bufB,ldb,0,
                                   0,
                                   bufC,N,0,params);
		 clFlush(params.queue);
		DWORD threadId;
		HANDLE threadHandle;
		BlasParams myParams;
		if(cpuPart>0) {
			//myParams.order = CblasRowMajor;
			if(transA) myParams.transA = 'T';
			else myParams.transA = 'N';
			if(transB) myParams.transB = 'T';
			else myParams.transB = 'N';
			myParams.M = M*cpuPart/(cpuPart+gpuPart);
			myParams.N = N;
			myParams.K = K;
			myParams.alpha = 1;
			if(transA){
				myParams.A = &A[M*gpuPart/(cpuPart+gpuPart)];
			}else{
				myParams.A = &A[M*N*gpuPart/(cpuPart+gpuPart)];
			}
			myParams.lda = M;
			myParams.B = B;
			myParams.ldb = N;
			myParams.beta = 0;
			myParams.C = &C[M*N*gpuPart/(cpuPart+gpuPart)];
			myParams.ldc = N;
			//threadHandle = CreateThread(0,0,BlasFun,&myParams,0,&threadId);
			//WaitForSingleObject(threadHandle,INFINITE);
			BlasFun(&myParams);
		}
        //clFinish(params.queue);
		//clWaitForEvents(1,&evt);
		
    }
	clFinish(q);
	 rt.stop();
	 	cout<<"Buffer creation time "<<bufTimer.getDiff()<<endl;
	 tdiff = rt.getDiff()/niters;
	  //cout<<"Time "<<tdiff<<endl;
	//clEnqueueUnmapMemObject(q,bufC,ptr,0,NULL,NULL);
	clEnqueueReadBuffer(q,bufC,CL_TRUE,0,sizeC*gpuPart/(cpuPart+gpuPart),&C[N*cpuPart*M/(cpuPart+gpuPart)],0,NULL,NULL);
	//ptr = clEnqueueMapBuffer(q,bufC,CL_TRUE,CL_MAP_READ,0,sizeC*gpuPart/(cpuPart+gpuPart),0,NULL,NULL,NULL);
	clFinish(q);
	delete gemm;
	double error = 0.0;
	/*switch(pattern){
		case 1:
			error = verifyPattern1<float>(A,B,C,N,N,N);
			break;
		case 2:
			error = verifyPattern2<float>(A,B,C,N,N,N);
			break;
	}*/
//	clEnqueueUnmapMemObject(q,bufC,ptr,0,NULL,NULL);
	//clFinish(q);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseCommandQueue(q);
	clGetContextInfo(ctx,CL_CONTEXT_REFERENCE_COUNT,sizeof(cl_uint),&refCount,NULL);
	cout<<"Reference count of context "<<refCount<<endl;
	clReleaseContext(ctx);
    cout<<"Max error "<<error<<" GFlops "<<(2.0e-9*M*N*K/tdiff)<<" time "<<tdiff<<endl;
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
}
#endif

struct TestGemmSingle{
	typedef float realtype;
	typedef RaijinSgemm gemmtype;
};

struct TestGemmDouble{
	typedef double realtype;
	typedef RaijinDgemm gemmtype;
};

int main(int argc, char **argv){
	cl_platform_id platforms[10];
	cl_uint num_platforms;
	clGetPlatformIDs(10,platforms,&num_platforms);


	if(argv[1][0]=='L'){
		for(int i=0;i<num_platforms;i++){
			cl_platform_id platform = platforms[i];
			char platname[200];
			clGetPlatformInfo(platform,CL_PLATFORM_NAME,sizeof(platname),platname,NULL);
			cout<<"Platform "<<i<<" :"<<platname<<endl;
			cl_device_id dvcs[20];
			cl_uint num_dvcs;
			clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,20,dvcs,&num_dvcs);
			for(int j=0;j<num_dvcs;j++){
				cl_device_id dvc = dvcs[j];
				char dvcname[200];
				clGetDeviceInfo(dvc,CL_DEVICE_NAME,sizeof(dvcname),dvcname,NULL);
				cl_uint vendorid;
				clGetDeviceInfo(dvc,CL_DEVICE_VENDOR_ID,sizeof(vendorid),&vendorid,NULL);
				cout<<"Device "<<j<<":"<<dvcname<<" from vendor "<<vendorid<<endl;
				//clGetDeviceInfo(dvc,CL_DEVICE_OPENCL_VERSION_MAJOR,
#ifdef RAIJIN_EXPERIMENTAL
				cl_device_fp_config fpconfig;
				clGetDeviceInfo(dvc,CL_DEVICE_DOUBLE_FP_CONFIG,sizeof(fpconfig),&fpconfig,NULL);
				cout<<"Double precision support:";
				if(fpconfig==0){
					cout<<" None";
				}
				if(fpconfig & CL_FP_DENORM){
					cout<<" CL_FP_DENORM";
				}
				if(fpconfig & CL_FP_INF_NAN){
					cout<<" CL_FP_INF_NAN";
				}
				if(fpconfig & CL_FP_ROUND_TO_NEAREST){
					cout<<" CL_FP_ROUND_TO_NEAREST";
				}
				if(fpconfig & CL_FP_ROUND_TO_ZERO){
					cout<" CL_FP_ROUND_TO_ZERO";
				}
				if(fpconfig & CL_FP_ROUND_TO_INF){
					cout<<" CL_FP_ROUND_TO_INF";
				}
				if(fpconfig & CL_FP_FMA){
					cout<<" CL_FP_FMA";
				}
				if(fpconfig & CL_FP_SOFT_FLOAT){
					cout<<" CL_FP_SOFT_FLOAT";
				}
				cout<<endl;
#endif
			}
		}

	}else if(argv[1][0]=='D'){
		int platid = atoi(argv[2]);
		int devid = atoi(argv[3]);
		int size = atoi(argv[4]);
		char dtype = argv[5][0];
		char pattern = atoi(argv[6]);
        bool transA = false;
        bool transB = false;
        if(argc>7) transA = atoi(argv[7])>0;
        if(argc>8) transB = atoi(argv[8])>0;
		if(argc>9) cpuPart = atoi(argv[9]);
		else cpuPart = 8;
		gpuPart = 16 - cpuPart;

		cout<<"CPU part "<<cpuPart<<" GPU part "<<gpuPart<<endl;
		if(platid>=num_platforms || platid<0){
			cout<<"Error. Platform ID not in range"<<endl;
			exit(1);
		}

		cl_platform_id platform = platforms[platid];
		char platname[200];
		clGetPlatformInfo(platform,CL_PLATFORM_NAME,sizeof(platname),platname,NULL);

		cl_device_id dvcs[20];
		cl_uint num_dvcs;
		clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,20,dvcs,&num_dvcs);
		if(devid>=num_dvcs || devid<0){
			cout<<"Error. Device ID not in range"<<endl;
			exit(1);
		}

		char dvcname[200];
		cl_device_id dvc = dvcs[devid];
		clGetDeviceInfo(dvc,CL_DEVICE_NAME,sizeof(dvcname),dvcname,NULL);
		cl_uint vendorid;
		clGetDeviceInfo(dvc,CL_DEVICE_VENDOR_ID,sizeof(vendorid),&vendorid,NULL);
		cout<<"Platform: "<<platname<<endl;
		cout<<"Device: "<<dvcname<<" from vendor "<<vendorid<<endl;
		if(dtype=='d'){
            testGemm<TestGemmDouble>(dvc,size,pattern,transA,transB);
		}else if(dtype=='s'){
			for(int i=1;i<=5;i++){
#ifdef RAIJIN_EXPERIMENTAL
            testPartSgemm(dvc,size,pattern,transA,transB);
#else
				testGemm<TestGemmSingle>(dvc,size,pattern,transA,transB);
#endif

			}
        }else if(dtype=='z'){
            testGemmComplex<TestGemmZ>(dvc,size,transA,transB,true);
        }else if(dtype='c'){
            testGemmComplex<TestGemmC>(dvc,size,transA,transB,true);
        }

	}

}
