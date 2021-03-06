#include "raijin.hpp"
#include <sstream>
#include <fstream>
#include "json/json.h"
using namespace std;
using namespace RaijinCL;

Json::Value RaijinTransOpt::toJson() const{
    Json::Value val(Json::objectValue);
    val["lx"] = lx;
    val["ly"] = ly;
    val["kernel"] = kernel;
    val["isImg"] = isImg;
    val["simdw"] = simdw;
    return val;
}

RaijinTransOpt::RaijinTransOpt(const Json::Value &val):kernel(){
    lx = val.get("lx",0).asInt();
    ly = val.get("ly",0).asInt();
    isImg = val.get("isImg",false).asBool();
    simdw = val.get("simdw",0).asInt();
    kernel = val.get("kernel","").asString();
}


static int simdToIdx(int simdw){
    switch(simdw){
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    default:
        return 4;
    }
}

/*static int flagToIndex(int val){
    return (val>0)? 0:1;
}*/

static bool buildKernel(cl_device_id dvc,cl_context ctx,string ksrc,cl_kernel *res){
    const char *kernel = ksrc.c_str();
    const size_t klen = ksrc.length();
    cl_int errcode1,errcode2,errcode3;
    cl_program prg = clCreateProgramWithSource(ctx,1,&kernel,&klen,&errcode1);
    errcode2 = clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel clkrnl = clCreateKernel(prg,"transKernel",&errcode3);
    if(errcode1==CL_SUCCESS && errcode2==CL_SUCCESS && errcode3==CL_SUCCESS){
        *res = clkrnl;
        //cout<<"Succesfully built kernel"<<endl;
        return true;
    }else{
		cout<<kernel<<endl;
        cout<<"ERROR: kernel did not build correctly "<<errcode1<<" "<<errcode2<<" "<<errcode3<<endl;
        return false;
    }
}

static void initTransData(cl_device_id dvc,cl_context ctx,RaijinTransOpt sparams[][4],bool sinit[][4],cl_kernel skernels[][4],string spath){
    //cout<<"Initializing trans data from path "<<spath<<endl;
    ifstream ifiles(spath.c_str());
	if(!ifiles.is_open() && ifiles.good()) return;
    int lx,ly;
    string line;
    Json::Value root;
    if(ifiles.is_open()) ifiles>>root;
    Json::Value array = root["cases"];
    for(int k=0;k<array.size();k++){
        Json::Value obj = array[k];
        RaijinTransOpt opt(obj);
        int i = opt.isImg ? 0 : 1;
        int j = simdToIdx(opt.simdw);
        sparams[i][j] = opt;
        bool built = buildKernel(dvc,ctx,opt.kernel,&skernels[i][j]);
        if(built) sinit[i][j] = true;
    }
}

RaijinTranspose::RaijinTranspose(cl_device_id device, cl_context context):dvc(device),ctx(context){
    clRetainContext(context);
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
            sinit[i][j] = false;
            dinit[i][j] = false;
            cinit[i][j] = false;
            zinit[i][j] = false;
            sgkernels[i][j] = NULL;
            dgkernels[i][j] = NULL;
            cgkernels[i][j] = NULL;
            zgkernels[i][j] = NULL;
			skernels[i][j] = NULL;
			dkernels[i][j] = NULL;
			ckernels[i][j] = NULL;
			zkernels[i][j] = NULL;
        }
    }
    string spath = raijinGetProfileFileName(dvc,"strans");
    initTransData(dvc,ctx,sparams,sinit,skernels,spath);
    cout<<"Generating generic kernel versions for strans"<<endl;
    const int simdw[] = {1,2,4,8};
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
            bool toImg = (i==0)? true:false;
            string res;
            bool ret = createRealTrans(simdw[j],1,1,false,toImg,false,res,false);
            if(ret){
                cl_kernel krnl;
				if(simdw[j]==4) cout<<res<<endl;
                ret = buildKernel(dvc,ctx,res,&krnl);
                if(ret){
                    sgkernels[i][j] = krnl;
                }
            }
        }
    }

    string cpath = raijinGetProfileFileName(dvc,"ctrans");
    initTransData(dvc,ctx,cparams,cinit,ckernels,cpath);
	for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            bool toImg = i==0? true:false;
            string res;
            bool ret = createComplexTrans(simdw[j],1,1,false,toImg,false,res,false);
            if(ret){
                cl_kernel krnl;
                ret = buildKernel(dvc,ctx,res,&krnl);
                if(ret){
                    cgkernels[i][j] = krnl;
                }
            }
        }
    }

    if(supportsFp64(dvc)){
        string dpath = raijinGetProfileFileName(dvc,"dtrans");
        initTransData(dvc,ctx,dparams,dinit,dkernels,dpath);
        cout<<"Generating generic kernel versions for dtrans"<<endl;
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                bool toImg = i==0? true:false;
                string res;
                bool ret = createRealTrans(simdw[j],1,1,false,toImg,true,res,false);
                if(ret){
                    cl_kernel krnl;
                    ret = buildKernel(dvc,ctx,res,&krnl);
                    if(ret){
                        dgkernels[i][j] = krnl;
                    }
                }
            }
        }
        string zpath = raijinGetProfileFileName(dvc,"ztrans");
        initTransData(dvc,ctx,zparams,zinit,zkernels,zpath);
		for(int i=0;i<2;i++){
			for(int j=0;j<2;j++){
				bool toImg = i==0? true:false;
				string res;
				bool ret = createComplexTrans(simdw[j],1,1,false,toImg,true,res,false);
				if(ret){
					cl_kernel krnl;
					ret = buildKernel(dvc,ctx,res,&krnl);
					if(ret){
						zgkernels[i][j] = krnl;
					}
				}
			}
		}
    }
}

static void releaseKrnl(cl_kernel krnl){
    if(krnl==NULL) return;
    //cl_program prg;
    //clGetKernelInfo(krnl,CL_KERNEL_PROGRAM,sizeof(cl_program),&prg,NULL);
    clReleaseKernel(krnl);
    //clReleaseProgram(prg);
}

RaijinTranspose::~RaijinTranspose(){
	//cout<<"transpose destructor"<<endl;
    for(int i=0;i<2;i++){
        for(int j=0;j<4;j++){
			//cout<<"Iteration "<<i<<","<<j<<endl;
            if(sinit[i][j]) releaseKrnl(skernels[i][j]);
            releaseKrnl(sgkernels[i][j]);

            if(dinit[i][j]) releaseKrnl(dkernels[i][j]);
            releaseKrnl(dgkernels[i][j]);

            if(cinit[i][j]) releaseKrnl(ckernels[i][j]);
            releaseKrnl(cgkernels[i][j]);

            if(zinit[i][j]) releaseKrnl(zkernels[i][j]);
            releaseKrnl(zgkernels[i][j]);
        }
    }
    clReleaseContext(ctx);
}

static cl_event applyTrans(cl_command_queue q,const RaijinTransOpt& opt,bool wasInited,cl_kernel genKernel,cl_kernel optKernel,
                           cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int nOutRows = endCol-startCol;
    const int nOutCols = (endRow-startRow)/simd;
    cl_kernel krnl;
    size_t gsize[2],lsizeOpt[2];
    size_t *lsize = NULL;
    if(wasInited && nOutRows%opt.lx==0 && nOutCols%opt.ly==0){
        cout<<"applyTrans: Applying opt kernel"<<endl;
        krnl = optKernel;
        lsizeOpt[1] = opt.lx;
        lsizeOpt[0] = opt.ly;
        lsize = lsizeOpt;
    }else{
        //apply generic kernel
        cout<<"applyTrans: Applying generic kernel"<<endl;
        krnl = genKernel;
    }
    gsize[1] = nOutRows;
    gsize[0] = nOutCols;
    size_t inbufsize,outbufsize;
    clGetMemObjectInfo(input,CL_MEM_SIZE,sizeof(inbufsize),&inbufsize,NULL);
    clGetMemObjectInfo(output,CL_MEM_SIZE,sizeof(outbufsize),&outbufsize,NULL);
    cout<<"applyTrans: In buffer size: "<<inbufsize<<" Out buffer size: "<<outbufsize<<endl;
    cout<<"applyTrans: gsize "<<nOutCols<<" "<<nOutRows<<endl;
    cout<<"applyTrans: "<<startRow<<" "<<endRow<<" "<<startCol<<" "<<endCol<<" "<<lda<<endl;
    cl_int kcode0,kcode1,kcode2,kcode3,kcode4,kcode5,kcode6;
    kcode0 = clSetKernelArg(krnl,0,sizeof(cl_mem),&input);
    kcode1 = clSetKernelArg(krnl,1,sizeof(cl_mem),&output);
    kcode2 = clSetKernelArg(krnl,2,sizeof(cl_int),&startRow);
    kcode3 = clSetKernelArg(krnl,3,sizeof(cl_int),&endRow);
    kcode4 = clSetKernelArg(krnl,4,sizeof(cl_int),&startCol);
    kcode5 = clSetKernelArg(krnl,5,sizeof(cl_int),&endCol);
    kcode6 = clSetKernelArg(krnl,6,sizeof(cl_int),&lda);
    cl_event event;
    cl_int retcode = clEnqueueNDRangeKernel(q,krnl,2,NULL,gsize,lsize,0,NULL,&event);
    if(retcode!=CL_SUCCESS){
        cout<<"applyTrans: "<<kcode0<<" "<<kcode1<<" "<<kcode2<<" "<<kcode3<<" "<<kcode4<<" "<<kcode5<<" "<<kcode6<<endl;
        cout<<"applyTrans: return code "<<retcode<<endl;
    }
    return event;
}

cl_event RaijinTranspose::stransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = sparams[1][j];
    return applyTrans(q,opt,sinit[1][j],sgkernels[1][j],skernels[1][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}

cl_event RaijinTranspose::dtransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = dparams[1][j];
    return applyTrans(q,opt,dinit[1][j],dgkernels[1][j],dkernels[1][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}

cl_event RaijinTranspose::ctransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = cparams[1][j];
    return applyTrans(q,opt,cinit[1][j],cgkernels[1][j],ckernels[1][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}

cl_event RaijinTranspose::ztransToBuf(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = zparams[1][j];
    return applyTrans(q,opt,zinit[1][j],zgkernels[1][j],zkernels[1][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}

cl_event RaijinTranspose::stransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = sparams[0][j];
    return applyTrans(q,opt,sinit[0][j],sgkernels[0][j],skernels[0][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}



cl_event RaijinTranspose::dtransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = dparams[0][j];
    return applyTrans(q,opt,dinit[0][j],dgkernels[0][j],dkernels[0][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}


cl_event RaijinTranspose::ctransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = cparams[0][j];
    return applyTrans(q,opt,cinit[0][j],cgkernels[0][j],ckernels[0][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}



cl_event RaijinTranspose::ztransToImg(cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    const int j = simdToIdx(simd);
    const RaijinTransOpt opt = zparams[0][j];
    return applyTrans(q,opt,zinit[0][j],zgkernels[0][j],zkernels[0][j],
                      input,output,simd,startRow,endRow,startCol,endCol,lda);
}


bool RaijinTranspose::createRealTrans(int simdw,int lx,int ly,bool useLocalMem,bool toImg,bool isDouble,string& res,bool insertAttrib){
    /* Each work-group writes lx rows and ly columns, where each column is of width simdw.
    Thus each work group must read ly*simdw rows and lx columns*/
    if(useLocalMem && lx!=ly) return false;
    int ncomp = simdw;
    if(isDouble) ncomp*=2;
    if(toImg && ncomp>4) return false;
    stringstream ss;
    const string dtype = (isDouble) ? "double" : "float";
    if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}
    if(simdw==1) ss<<"typedef "<<dtype<<" "<<dtype<<"1;"<<endl;

    if(insertAttrib) ss<<"__attribute__((reqd_work_group_size("<<ly<<","<<lx<<",1)))"<<endl;
    ss<<"__kernel void transKernel(__global const "<<dtype<<" *input,";

    if(toImg){
        ss<<"__write_only image2d_t output";
    }else{
        ss<<"__global "<<dtype<<simdw<<" *output";
    }
    ss<<",int rstart,int rend,int cstart,int cend,int lda){"<<endl;
    ss<<"const unsigned int i = get_global_id(1);"<<endl;
    ss<<"const unsigned int j = get_global_id(0);"<<endl;
    for(int i=0;i<simdw;i++) ss<<dtype<<" inval"<<i<<";"<<endl;
    if(useLocalMem){
        ss<<"const int lidx = get_local_id(1);"<<endl;
        ss<<"const int lidy = get_local_id(0);"<<endl;
        ss<<"const int lx = "<<lx<<";"<<endl;
        ss<<"const int ly = "<<ly<<";"<<endl;
        ss<<"const int ostartx = get_group_id(1)*lx;"<<endl;
        ss<<"const int ostarty = get_group_id(0)*ly;"<<endl;
        ss<<"__local "<<dtype<<" lds["<<lx<<"]["<<(ly*simdw+1)<<"];"<<endl;
        for(int i=0;i<simdw;i++){
            ss<<"lds[lidy][lidx+"<<lx<<"*"<<i<<"] = input[(rstart+ostarty*"<<simdw<<"+lidx+"<<lx<<"*"<<i<<")*lda + cstart+ostartx+lidy];"<<endl;
        }
        ss<<"barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i<<" = lds[lidx][lidy*"<<simdw<<"+"<<i<<"];"<<endl;
        }
    }else{
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i<<" = input[(j*"<<simdw<<"+"<<i<<"+rstart)*lda + i+cstart];"<<endl;
        }
    }
    if(toImg){
        if(isDouble){
            if(simdw==1){
                ss<<"int4 outputVal;\n";
                ss<<"outputVal.xy = as_int2(inval0); outputVal.z = 0; outputVal.w = 1;"<<endl;
            }else if(simdw==2){
                ss<<"double2 outputVal;"<<endl;
                ss<<"outputVal.x = inval0; outputVal.y = inval1;"<<endl;
            }
            ss<<"write_imagei(output,(int2)(j,i),as_int4(outputVal));"<<endl;
        }else{
            ss<<"write_imagef(output,(int2)(j,i),(float4)(";
            for(int i=0;i<4;i++){
                if(i<simdw) ss<<"inval"<<i;
                else ss<<0;
                if(i<3) ss<<",";
            }
            ss<<"));"<<endl;
        }
    }else{
        ss<<"output[i*(rend-rstart)/"<<simdw<<"+j] = ("<<dtype<<simdw<<")(";
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i;
            if(i<(simdw-1)) ss<<",";
        }
        ss<<");"<<endl;
    }
    ss<<"}"<<endl;
    res = ss.str();
    return true;
}

bool RaijinTranspose::createComplexTrans(int simdw,int lx,int ly,bool useLocalMem,bool toImg,bool isDouble,std::string& res,bool insertAttrib){
	 /* Each work-group writes lx rows and ly columns, where each column is of width simdw.
    Thus each work group must read ly*simdw rows and lx columns*/
    if(useLocalMem && lx!=ly) return false;
    int ncomp = simdw*2;
    if(isDouble) ncomp*=2;
    if(toImg && ncomp>4) return false;
    stringstream ss;
    const string dtype = (isDouble) ? "double" : "float";

	if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}

    if(insertAttrib) ss<<"__attribute__((reqd_work_group_size("<<ly<<","<<lx<<",1)))"<<endl;
    ss<<"__kernel void transKernel(__global const "<<dtype<<"2 *input,";

    if(toImg){
        ss<<"__write_only image2d_t output";
    }else{
        ss<<"__global "<<dtype<<(simdw*2)<<" *output";
    }
    ss<<",int rstart,int rend,int cstart,int cend,int lda){"<<endl;
    ss<<"const unsigned int i = get_global_id(1);"<<endl;
    ss<<"const unsigned int j = get_global_id(0);"<<endl;
    for(int i=0;i<simdw;i++) ss<<dtype<<"2 inval"<<i<<";"<<endl;
    if(useLocalMem){
        ss<<"const int lidx = get_local_id(1);"<<endl;
        ss<<"const int lidy = get_local_id(0);"<<endl;
        ss<<"const int lx = "<<lx<<";"<<endl;
        ss<<"const int ly = "<<ly<<";"<<endl;
        ss<<"const int ostartx = get_group_id(1)*lx;"<<endl;
        ss<<"const int ostarty = get_group_id(0)*ly;"<<endl;
        ss<<"__local "<<dtype<<"2 lds["<<lx<<"]["<<(ly*simdw+1)<<"];"<<endl;
        for(int i=0;i<simdw;i++){
            ss<<"lds[lidy][lidx+"<<lx<<"*"<<i<<"] = input[(ostarty*"<<simdw<<"+lidx+"<<lx<<"*"<<i<<"+rstart)*lda + ostartx+lidy+cstart];"<<endl;
        }
        ss<<"barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i<<" = lds[lidx][lidy*"<<simdw<<"+"<<i<<"];"<<endl;
        }
    }else{
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i<<" = input[(rstart+j*"<<simdw<<"+"<<i<<")*lda + i+cstart];"<<endl;
        }
    }
    if(toImg){
        if(isDouble){
            if(simdw==1){
                ss<<"int4 outputVal = as_int4(inval0);"<<endl;
				ss<<"write_imagei(output,(int2)(j,i),outputVal);"<<endl;
            }
        }else{
			if(simdw==1) ss<<"float2 inval1; inval1.x = 0; inval1.y = 1;\n";
            ss<<"write_imagef(output,(int2)(j,i),(float4)(inval0,inval1));\n";
        }
    }else{
        ss<<"output[i*(rend-rstart)/"<<simdw<<"+j] = ("<<dtype<<(simdw*2)<<")(";
        for(int i=0;i<simdw;i++){
            ss<<"inval"<<i;
            if(i<(simdw-1)) ss<<",";
        }
        ss<<");"<<endl;
    }
    ss<<"}"<<endl;
    res = ss.str();
	//cout<<"Created complex kernel "<<res<<endl;
    return true;
}

template <>
cl_event RaijinCL::raijinCopyToBuf<cl_float>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->scopyToBuf(q,input,output,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToBuf<cl_float2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->ccopyToBuf(q,input,output,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToBuf<cl_double>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->dcopyToBuf(q,input,output,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToBuf<cl_double2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->zcopyToBuf(q,input,output,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToImg<cl_float>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->scopyToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToImg<cl_double>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->dcopyToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToImg<cl_float2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->ccopyToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::raijinCopyToImg<cl_double2>(RaijinCopy *copyObj,cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return copyObj->zcopyToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

static cl_kernel createCopyKernel(bool isDouble,bool isComplex,cl_context ctx,cl_device_id dvc){
    stringstream ss;
    const string dtype = isDouble ? "double" : "float";
    const string elemtype = isComplex ? (dtype+"2") : dtype;

	if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}

    ss<<"__kernel void copyKernel(__global const "<<elemtype<<" *input,__global "<<elemtype<<" *output,";
    ss<<"int startRow,int endRow,int startCol,int endCol,int lda){"<<endl;
    ss<<"const unsigned int i = get_global_id(1); const unsigned int j = get_global_id(0);"<<endl;
    ss<<" output[i*(endCol-startCol) + j] = input[(i+startRow)*lda + (j+startCol)];"<<endl;
    ss<<"}"<<endl;
    string kernelstr = ss.str();
    //cout<<"createCopyKernel "<<kernelstr<<endl;
    const char *kernelcstr = kernelstr.c_str();
    size_t ksize = kernelstr.size();
	cl_int errcode1,errcode2,errcode3;
    cl_program prg = clCreateProgramWithSource(ctx,1,&kernelcstr,&ksize,&errcode1);
    errcode2 = clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"copyKernel",&errcode3);
	if(errcode1!=CL_SUCCESS || errcode2!=CL_SUCCESS || errcode3!=CL_SUCCESS){
		 cout<<"createCopyKernelImg "<<errcode1<<" "<<errcode2<<" "<<errcode3<<endl;
        size_t retbytes;
        clGetProgramBuildInfo(prg, dvc, CL_PROGRAM_BUILD_LOG, 0, NULL, &retbytes);
        char *buildlog = new char[retbytes+1];
        clGetProgramBuildInfo(prg,dvc,CL_PROGRAM_BUILD_LOG,retbytes,buildlog,NULL);
        cout << "Buildlog " << retbytes<<" "<<buildlog << endl;
        delete[] buildlog;
	}
    /* prg = clCreateProgramWithSource(ctx,1,&kernelcstr,&ksize,NULL);
    clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"copyKernel",NULL);*/
    return krnl;
}

static cl_kernel createCopyKernelImg(bool isDouble,bool isComplex,int simd,cl_context ctx,cl_device_id dvc){
    stringstream ss;
    const string dtype = isDouble ? "double" : "float";
    const string elemtype = isComplex ? (dtype+"2") : dtype;

	if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}

    int ncomp = simd;
    if(isDouble) ncomp*=2;
    if(isComplex) ncomp*=2;

    ss<<"__kernel void copyKernel(__global const "<<elemtype<<" *input,__write_only image2d_t output,";
    ss<<"int startRow,int endRow,int startCol,int endCol,int lda){"<<endl;
    ss<<"const unsigned int i = get_global_id(1); const unsigned int j = get_global_id(0);"<<endl;
    for(int i=0;i<simd;i++) ss<<" "<<elemtype<<" inval"<<i<<" = input[(i+startRow)*lda+startCol + j*"<<simd<<"+"<<i<<"];"<<endl;
    if(isDouble){
        if(isComplex) ss<<"double2 outVal = inval0;"<<endl;
        else{
            if(simd==1){
                ss<<"int4 outVal = (int4)(as_int2(inval0),0,1);";
            }else if(simd==2){
                ss<<"double2 outVal = (double2)(inval0,inval1);";
            }
        }
        ss<<" write_imagei(output,(int2)(j,i),as_int4(outVal));"<<endl;

    }else{
        ss<<"float4 outVal= (float4)(";
        for(int i=0;i<simd;i++){
            ss<<"inval"<<i;
            if(i<(simd-1) || (ncomp<4)) ss<<",";
        }
        for(int i=ncomp;i<4;i++){
            if(i==3) ss<<1;
            else ss<<0<<",";
        }
        ss<<");"<<endl;
        ss<<" write_imagef(output,(int2)(j,i),outVal);"<<endl;
    }

    ss<<"}"<<endl;
    string kernelstr = ss.str();
	if(simd==4) cout<<kernelstr<<endl;
    const char *kernelcstr = kernelstr.c_str();
    size_t ksize = kernelstr.size();
	cl_int errcode1,errcode2,errcode3;
    cl_program prg = clCreateProgramWithSource(ctx,1,&kernelcstr,&ksize,&errcode1);
    errcode2 = clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"copyKernel",&errcode3);
	if(errcode1!=CL_SUCCESS || errcode2!=CL_SUCCESS || errcode3!=CL_SUCCESS){
		 cout<<"createCopyKernelImg "<<errcode1<<" "<<errcode2<<" "<<errcode3<<endl;
        size_t retbytes;
        clGetProgramBuildInfo(prg, dvc, CL_PROGRAM_BUILD_LOG, 0, NULL, &retbytes);
        char *buildlog = new char[retbytes+1];
        clGetProgramBuildInfo(prg,dvc,CL_PROGRAM_BUILD_LOG,retbytes,buildlog,NULL);
        cout << "Buildlog " << retbytes<<" "<<buildlog << endl;
        delete[] buildlog;
	}
    return krnl;
}

RaijinCopy::RaijinCopy(cl_context context, cl_device_id device):ctx(context),dvc(device){
    clRetainContext(ctx);
    sbuf = createCopyKernel(false,false,ctx,dvc);
    cbuf = createCopyKernel(false,true,ctx,dvc);
    int simd = 1;
    for(int i=0;i<3;i++) {
        if(simd<=4) simg[i] = createCopyKernelImg(false,false,simd,ctx,dvc);
        if(simd<=2) cimg[i] = createCopyKernelImg(false,true,simd,ctx,dvc);
        simd *= 2;
    }

    if(supportsFp64(dvc)){
        dbuf = createCopyKernel(true,false,ctx,dvc);
        zbuf = createCopyKernel(true,true,ctx,dvc);
        simd = 1;
        for(int i=0;i<2;i++) {
            if(simd<=2) dimg[i] = createCopyKernelImg(true,false,simd,ctx,dvc);
            if(simd<=1) zimg = createCopyKernelImg(true,true,simd,ctx,dvc);
            simd *= 2;
        }
    }

}

static cl_event dispatchCopyBuf(cl_command_queue q, cl_mem input, cl_mem output, int startRow, int endRow, int startCol, int endCol,
                                int lda,cl_kernel krnl,int simd=1){
   // cout<<"dispatchCopyBuf "<<startRow<<" "<<endRow<<" "<<startCol<<" "<<endCol<<" "<<lda<<" "<<simd<<endl;
	cl_int kcode0,kcode1,kcode2,kcode3,kcode4,kcode5,kcode6;
    kcode0 = clSetKernelArg(krnl,0,sizeof(cl_mem),&input);
    kcode1 = clSetKernelArg(krnl,1,sizeof(cl_mem),&output);
    kcode2 = clSetKernelArg(krnl,2,sizeof(cl_int),&startRow);
    kcode3 = clSetKernelArg(krnl,3,sizeof(cl_int),&endRow);
    kcode4 = clSetKernelArg(krnl,4,sizeof(cl_int),&startCol);
    kcode5 = clSetKernelArg(krnl,5,sizeof(cl_int),&endCol);
    kcode6 = clSetKernelArg(krnl,6,sizeof(cl_int),&lda);
    size_t gsize[] = {(endCol-startCol)/simd,(endRow-startRow)};
    cl_event evt;
    //cl_event *waitList = NULL;
    //if(rp.num_events>0) waitList = rp.waitEvents;
    cl_int rcode = clEnqueueNDRangeKernel(q,krnl,2,NULL,gsize,NULL,0,NULL,&evt);
    if(rcode!=CL_SUCCESS){
        cout<<"dispatchCopyBuf: "<<kcode0<<" "<<kcode1<<" "<<kcode2<<" "<<kcode3<<" "<<kcode4<<" "<<kcode5<<" "<<kcode6<<endl;
        cout<<"dispatchCopyBuf: return code "<<rcode<<endl;
    }
    return evt;
}

cl_event RaijinCopy::scopyToBuf(cl_command_queue q, cl_mem input, cl_mem output, int startRow, int endRow, int startCol, int endCol, int lda){
    //cout<<"raijinCopy::scopyToBuf"<<endl;
   return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,sbuf,1);
}

cl_event RaijinCopy::dcopyToBuf(cl_command_queue q, cl_mem input, cl_mem output, int startRow, int endRow, int startCol, int endCol, int lda){
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,dbuf,1);
}

cl_event RaijinCopy::ccopyToBuf(cl_command_queue q, cl_mem input, cl_mem output, int startRow, int endRow, int startCol, int endCol, int lda){
    cout<<"Called ccopyTobuf"<<endl;
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,cbuf,1);
}

cl_event RaijinCopy::zcopyToBuf(cl_command_queue q, cl_mem input, cl_mem output, int startRow, int endRow, int startCol, int endCol, int lda){
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,zbuf,1);
}

cl_event RaijinCopy::scopyToImg(cl_command_queue q, cl_mem input, cl_mem output, int simd, int startRow, int endRow, int startCol, int endCol, int lda){
    cl_kernel krnl;
    if(simd==1) krnl = simg[0];
    if(simd==2) krnl = simg[1];
    if(simd==4) krnl = simg[2];
	//scout<<"RaijinCopy::scopyToImg: "<<simd<<" "<<startRow<<" "<<endRow<<" "<<startCol<<" "<<endCol<<endl;
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,krnl,simd);
}

cl_event RaijinCopy::dcopyToImg(cl_command_queue q, cl_mem input, cl_mem output, int simd, int startRow, int endRow, int startCol, int endCol, int lda){
    cl_kernel krnl;
    if(simd==1) krnl = dimg[0];
    if(simd==2) krnl = dimg[1];
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,krnl,simd);
}

cl_event RaijinCopy::ccopyToImg(cl_command_queue q, cl_mem input, cl_mem output, int simd, int startRow, int endRow, int startCol, int endCol, int lda){
    cl_kernel krnl;
    if(simd==1) krnl = cimg[0];
    if(simd==2) krnl = cimg[1];
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,krnl,simd);
}

cl_event RaijinCopy::zcopyToImg(cl_command_queue q, cl_mem input, cl_mem output, int simd, int startRow, int endRow, int startCol, int endCol, int lda){
    return dispatchCopyBuf(q,input,output,startRow,endRow,startCol,endCol,lda,zimg);
}

RaijinCopy::~RaijinCopy(){
    clReleaseKernel(sbuf);
    clReleaseKernel(cbuf);
    for(int i=0;i<3;i++) clReleaseKernel(simg[i]);
    for(int i=0;i<2;i++) clReleaseKernel(cimg[i]);
    if(supportsFp64(dvc)){
        clReleaseKernel(dbuf);
        clReleaseKernel(zbuf);
        for(int i=0;i<2;i++) clReleaseKernel(dimg[i]);
        clReleaseKernel(zimg);
    }
    clReleaseContext(ctx);
}


template <>
cl_event RaijinCL::transToBuf<cl_float>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return trans->stransToBuf(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToBuf<cl_double>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return trans->dtransToBuf(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToBuf<cl_float2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return trans->ctransToBuf(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToBuf<cl_double2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return trans->ztransToBuf(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}


template <>
cl_event RaijinCL::transToImg<cl_float>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
    return trans->stransToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToImg<cl_double>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
     return trans->dtransToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToImg<cl_float2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
     return trans->ctransToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

template <>
cl_event RaijinCL::transToImg<cl_double2>(RaijinTranspose *trans,
                    cl_command_queue q,cl_mem input,cl_mem output,int simd,int startRow,int endRow,int startCol,int endCol,int lda){
     return trans->ztransToImg(q,input,output,simd,startRow,endRow,startCol,endCol,lda);
}

static cl_kernel getScaleKernel(cl_context ctx,cl_device_id dvc,bool isDouble,bool isComplex){
    stringstream ss;

	if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}
    string dtype = isDouble ? "double" : "float";
    if(isComplex){
        ss<<"__kernel void scaleKernel(__global "<<dtype<<"2 *C,int ldc,"<<dtype<<"2 beta){"<<endl;
        ss<<"int i = get_global_id(1); int j = get_global_id(0);\n";
        ss<<dtype<<"2 inval = C[i*ldc + j];"<<endl;
        ss<<dtype<<"2 outval;"<<endl;
        ss<<"outval.x = (inval.x*beta.x - inval.y*beta.y); outval.y = (inval.x*beta.y + inval.y*beta.x);"<<endl;
        ss<<"C[i*ldc + j] = outval;"<<endl;
    }else{
        ss<<"__kernel void scaleKernel(__global "<<dtype<<" *C,int ldc,"<<dtype<<" beta){"<<endl;
        ss<<"int i = get_global_id(1); int j = get_global_id(0);\n";
        ss<<"C[i*ldc + j] *= beta;"<<endl;
    }
    ss<<"}"<<endl;
    const string kernelString = ss.str();
    const char *kernelCharString = kernelString.c_str();
    const size_t len = kernelString.length();
    cl_int bldcode1,bldcode2,bldcode3;
    cl_program prg = clCreateProgramWithSource(ctx,1,&kernelCharString,&len,&bldcode1);
    bldcode2 = clBuildProgram(prg,1,&dvc,"",NULL,NULL);
    cl_kernel krnl = clCreateKernel(prg,"scaleKernel",&bldcode3);
    if(bldcode1!=CL_SUCCESS || bldcode2!=CL_SUCCESS | bldcode3!=CL_SUCCESS){
        cout<<"applyScale "<<bldcode1<<" "<<bldcode2<<" "<<bldcode3<<endl;
        size_t retbytes;
        clGetProgramBuildInfo(prg, dvc, CL_PROGRAM_BUILD_LOG, 0, NULL, &retbytes);
        char *buildlog = new char[retbytes+1];
        clGetProgramBuildInfo(prg,dvc,CL_PROGRAM_BUILD_LOG,retbytes,buildlog,NULL);
        cout << "Buildlog " << retbytes<<" "<<buildlog << endl;
        delete[] buildlog;
    }
    return krnl;
}

RaijinScale::RaijinScale(cl_context context, cl_device_id device):ctx(context),dvc(device){
    //clRetainContext(ctx);
    skrnl = getScaleKernel(ctx,dvc,false,false);
    ckrnl = getScaleKernel(ctx,dvc,false,true);
    if(supportsFp64(dvc)){
        dkrnl = getScaleKernel(ctx,dvc,true,false);
        zkrnl = getScaleKernel(ctx,dvc,true,true);
    }
}

template <typename T>
cl_event dispatchScaleKernel(cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,T beta,cl_kernel krnl){
    clSetKernelArg(krnl,0,sizeof(cl_mem),&C);
    clSetKernelArg(krnl,1,sizeof(cl_int),&ldc);
    clSetKernelArg(krnl,2,sizeof(T),&beta);
    cl_event evt;
    size_t gsize[] = {N,M};
    cl_int retcode = clEnqueueNDRangeKernel(q,krnl,2,NULL,gsize,NULL,0,NULL,&evt);
    if(retcode!=CL_SUCCESS){
        cout<<"dispatchScaleKernel: "<<retcode<<endl;
    }
    return evt;
}

cl_event RaijinScale::sscale(cl_command_queue q, cl_mem C, size_t M, size_t N, cl_int ldc, cl_float beta){
    return dispatchScaleKernel<cl_float>(q,C,M,N,ldc,beta,skrnl);
}

cl_event RaijinScale::dscale(cl_command_queue q, cl_mem C, int M, int N, int ldc, cl_double beta){
    return dispatchScaleKernel<cl_double>(q,C,M,N,ldc,beta,dkrnl);
}

cl_event RaijinScale::cscale(cl_command_queue q, cl_mem C, int M, int N, int ldc, cl_float2 beta){
    cout<<"Called scale with param "<<beta.s[0]<<" "<<beta.s[1]<<endl;
    return dispatchScaleKernel<cl_float2>(q,C,M,N,ldc,beta,ckrnl);
}

cl_event RaijinScale::zscale(cl_command_queue q, cl_mem C, int M, int N, int ldc, cl_double2 beta){
    return dispatchScaleKernel<cl_double2>(q,C,M,N,ldc,beta,zkrnl);
}


template <>
cl_event RaijinCL::raijinScale<cl_float>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_float beta){
    return rs->sscale(q,C,M,N,ldc,beta);
}

template <>
cl_event RaijinCL::raijinScale<cl_float2>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_float2 beta){
    return rs->cscale(q,C,M,N,ldc,beta);
}

template <>
cl_event RaijinCL::raijinScale<cl_double>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_double beta){
    return rs->dscale(q,C,M,N,ldc,beta);
}

template <>
cl_event RaijinCL::raijinScale<cl_double2>(RaijinScale *rs,cl_command_queue q,cl_mem C,size_t M,size_t N,cl_int ldc,cl_double2 beta){
    return rs->zscale(q,C,M,N,ldc,beta);
}
