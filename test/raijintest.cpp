#include "raijin.hpp"
#include "raijin_complex.hpp"
#include <iostream>
#include <cmath>
#include "rtimer.hpp"
using namespace std;
using namespace RaijinCL;

template <typename T>
void pattern1(T *A, T *B,int M,int K,int N){
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			A[i*K+j] = (i*2.0)/K;
		}
	}
	for(int i=0;i<K;i++){
		for(int j=0;j<N;j++){
			B[i*N+j] = (j*2.0)/K;
		}
	}
}

template <typename T>
double verifyPattern1(T *A,T *B, T *C,int M,int N,int K){
	double sum = 0.0;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			double expected = K*(i*2.0/K)*(j*2.0/K);
			double read = C[i*N+j];
			//if(i==1) cout<<"Read "<<read<<endl;
            double diff = abs(read-expected);
            if(diff>1.0) {
                cout<<i<<" "<<j<<" "<<expected<<" "<<read<<" "<<endl;
                exit(1);
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
    RaijinParams params;
    params.num_events = 0;
    params.waitEvents = NULL;
    params.queue = q;
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
        cl_event evt = gemm->apply(RaijinCL::RaijinRowMajor,transA,transB,N,N,N,
                                                    alpha,bufA,N,0,bufB,N,0,beta,bufC,N,0,params);
        clWaitForEvents(1,&evt);
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
	RaijinParams params;
	params.num_events = 0;
	params.queue = q;
	params.waitEvents = NULL;
    const int niters = 5;
    double tdiff = 0;
    for(int i=0;i<niters;i++){
        RTimer rt;
        rt.start();
        cl_event evt = gemm->apply(RaijinRowMajor,transA,transB,
                                   M,N,K,
                                   1,bufA,K,0,
                                   bufB,N,0,
                                   0,
                                   bufC,N,0,params);
        clFlush(params.queue);
        clFinish(params.queue);
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
		if(platid>=num_platforms || platid<0){
			cout<<"Error. Platform ID not in range"<<endl;
			exit(1);
		}

		cl_platform_id platform = platforms[platid];
		char platname[200];
		clGetPlatformInfo(platform,CL_PLATFORM_NAME,sizeof(platname),platname,NULL);

		cl_device_id dvcs[20];
		cl_uint num_dvcs;
		clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,20,dvcs,&num_dvcs);
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
            testGemm<TestGemmSingle>(dvc,size,pattern,transA,transB);
        }else if(dtype=='z'){
            testGemmComplex<TestGemmZ>(dvc,size,transA,transB,true);
        }else if(dtype='c'){
            testGemmComplex<TestGemmC>(dvc,size,transA,transB,true);
        }

	}

}
