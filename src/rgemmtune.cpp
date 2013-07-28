#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "raijin.hpp"
#include "rtimer.hpp"
#include <CL/cl.h>

using namespace std;
using namespace RaijinCL;

//#define AMD_BUG


bool genKernelTNOff(int lsizex,
                         int lsizey,
                         int htile,
                         int wtile,
                         int ktile,
                         string dtype,
                         int simdwidth,
                         bool storea,
                         bool storeb,
                         int maxLocalMemElems,
                         int padding,
                         string& kernel,
                         bool useImageA = false,
                         bool useImageB = false){
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
    bool isDouble = (dtype.compare("double")==0) ? true:false;
#ifndef AMD_BUG
    if(isDouble && simdwidth>2 && (useImageA || useImageB)) return false;
    if(!isDouble && simdwidth>4 && (useImageA || useImageB)) return false;
#else
    if(isDouble && simdwidth!=2 && (useImageA || useImageB)) return false;
    if(!isDouble && simdwidth!=4 && (useImageA || useImageB)) return false;
#endif
    //const int unroll = ktile;
    if(storea){
        /*Number of rows of A per workgroup = ktile. Has to be divisble by lsizey. */
        if(ktile%lsizex!=0) return false;
        /*Number of columns of A per workgroup = htile*lsizex*/
        if((htile*lsizex)%(simdwidth*lsizey)!=0) return false;
    }else{
        if(htile%simdwidth!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = wtile*lsizey. Has to be divisble by lsizex*simdwidth */
        if(ktile%lsizex!=0) return false;
        if(((wtile*lsizey)%(lsizey*simdwidth))!=0) return false;
    }else{
        if(wtile%simdwidth!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;

    if(wtile%simdwidth!=0 || htile%simdwidth!=0) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += ktile*(htile*lsizex+padding);
    if(storeb) numLocalMemElems += ktile*(wtile*lsizey+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //if(ktile%unroll!=0) return false;
    //cout<<"Check 4 passed"<<endl;
    stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
    ss<<"const int htile ="<<htile<<";\n";
    ss<<"const int wtile ="<<wtile<<";\n";
    ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
    ss<<"const int lx = "<<lsizex<<";\n";
    ss<<"const int ly = "<<lsizey<<";\n";
    ss<<"const int i = get_global_id(1);"<<endl;
    ss<<"const int j = get_global_id(0);"<<endl;
    //ss<<"const int unroll = "<<unroll<<";\n";
    //ss<<"const int unsigned flat = get_local_id(0)+get_local_id(1)*ly;"<<endl;
    ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
    ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    /*ss<<"const unsigned int lidx = flat%lx;"<<endl;
    ss<<"const unsigned int lidy = (get_local_id(0)+get_local_id(1))%ly;"<<endl;*/
    ss<<"int k;"<<endl;
    if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(ktile)<<"]["<<((htile/simdwidth)*lsizex+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";
        }
    }
    ss<<"for(k=0;k<K;k+=ktile){"<<endl;
	ss<<" const uint gstartxA = get_group_id(1)*lx*htile;"<<endl;
	ss<<" const uint gstartxB = get_group_id(0)*ly*wtile;"<<endl;
	if(storea){
        for(int y=0;y<(ktile/lsizex);y++){
            for(int x=0;x<((htile*lsizex)/(simdwidth*lsizey));x++){
                ss<<" ldsA["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
                if(useImageA){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(A,sampler,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx )))";
                    }else{
                        ss<<"(myread_imagef(A,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx)))";
                    }
#ifndef AMD_BUG
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
#endif
                    ss<<";\n";
                }else{
                    ss<<"A[(k+"<<y<<"*lx+lidx)*(lda/simdwidth)+ gstartxA + lidy + "<<x<<"*ly];\n";
                }
            }
        }
    }
	if(storeb){
        for(int y=0;y<(ktile/lsizex);y++){
            for(int x=0;x<((wtile*lsizey)/(simdwidth*lsizey));x++){
                ss<<" ldsB["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(lidy + "<<x<<"*ly+gstartxB,k+"<<y<<"*lx +lidx)))";
                    }else{
                        ss<<"(myread_imagef(B,(int2)(lidy + "<<x<<"*ly+gstartxB,k+"<<y<<"*lx +lidx)))";
                    }
#ifndef AMD_BUG
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
#endif
                    ss<<";\n";
                }else{
                    ss<<"B[(k+"<<y<<"*lx+lidx)*(ldb/simdwidth)+ gstartxB + lidy + "<<x<<"*ly];\n";
                }
            }
        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    //ss<<" for(kk=0;kk<ktile;kk+=unroll){\n";
    ss<<"const int kk = 0;\n";

	for (int y = 0; y < ktile; y++) {
		for (int x = 0; x < wtile/simdwidth; x++) {
			ss << "  const " << dtype << simdwidth << " b" << x << "_" << y << " = ";
			if(storeb){
				ss << "ldsB[kk+"<<y<<"][ly*"<<x<<"+lidy];\n";
			}else{
				if(useImageB){
					if(isDouble){
						ss<<"as_double2( read_imagei(B,sampler,(int2)((get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy,k+kk+"<<y<<")))";
					}else{
						ss<<"( myread_imagef(B,(int2)((get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy,k+kk+"<<y<<")))";
					}
#ifndef AMD_BUG
					ss<<".s";
					for(int s=0;s<simdwidth;s++) ss<<s;
#endif
					ss<<";\n";
				}else{
					ss << "B[(k+kk+"<<y<<")*(ldb/simdwidth)+ (get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy];\n";
				}
			}
		}
	}

	for(int y=0;y<ktile;y++){
		for (int x = 0; x < htile/simdwidth; x++) {
			ss << "  const " << dtype << simdwidth << " a" << x << "_" << y << " = ";
			if(storea){
				ss << "ldsA[kk+"<<y<<"][lx*"<<x<<"+lidx];\n";
			}else{
				if(useImageA){
					if(isDouble) {
						ss<<"as_double2(read_imagei(A,sampler,(int2)((get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx,k+kk+"<<y<<")))";
					}else{
						ss<<"(myread_imagef(A,(int2)((get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx,k+kk+"<<y<<")))";
					}
#ifndef AMD_BUG
					ss<<".s";
					for(int s=0;s<simdwidth;s++) ss<<s;
#endif
					ss<<";\n";
				}else{
					ss << "A[(k+kk+"<<y<<")*(lda/simdwidth)+ (get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx];"<<endl;
				}
			}
		}
	}
	for (int x = 0; x < htile/simdwidth; x++) {
		for(int y=0;y<ktile;y++){
            for(int xoff=0;xoff<simdwidth;xoff++){
                    int row = x*simdwidth + xoff;
                    for(int w=0;w<wtile/simdwidth; w++){
                        if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
                        ss<<"  sum"<<row<<"_"<<w<<" += "<<"a"<<x<<"_"<<y;
                        if(simdwidth>1){
                            ss<<".s";
                            for(int m=0;m<simdwidth;m++) ss<<xoff;
                        }
                        ss<<"*b"<<w<<"_"<<y<<";\n";
                        if(isDouble){
                            ss<<"#else"<<endl;
                            ss<<"  sum"<<row<<"_"<<w<<" = fma( "<<"a"<<x<<"_"<<y;
                            if(simdwidth>1){
                                ss<<".s";
                                for(int m=0;m<simdwidth;m++) ss<<xoff;
                            }
                            ss<<",b"<<w<<"_"<<y<<",sum"<<row<<"_"<<w<<");\n";
                            ss<<"#endif"<<endl;
                        }
                    }
                }
            }
        }


    //ss<<" }\n";

    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    for (int i = 0; i < htile/simdwidth; i++) {
        for(int ii=0;ii<simdwidth;ii++){
            for (int j = 0; j < wtile/simdwidth; j++) {
                for(int jj=0;jj<simdwidth;jj++){

                    ss << "C[( (get_group_id(1)*htile+"<<i<<"*simdwidth)*lx + lidx*simdwidth+ "<<ii<<")*ldc + (get_group_id(0)*wtile + " << j << "*simdwidth)*ly + lidy*simdwidth+" << jj << "]";
                    ss << "= alpha*sum"<<(i*simdwidth+ii)<<"_"<<j;
                    if(simdwidth>1) ss<<".s"<<jj;
                    ss<<" + beta*";
                    ss << "C[( (get_group_id(1)*htile+"<<i<<"*simdwidth)*lx + lidx*simdwidth+ "<<ii<<")*ldc + (get_group_id(0)*wtile + " << j << "*simdwidth)*ly + lidy*simdwidth+" << jj << "]";
                    //ss << "C[(i*" << htile << "+ " << i << ")*ldc + (get_group_id(0)*wtile + " << j << "*simdwidth)*ly + lidy*simdwidth+" << jj << "]";
                    ss << ";" << endl;
                }
            }
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

bool genKernelTNCons(int lsizex,
                         int lsizey,
                         int htile,
                         int wtile,
                         int ktile,
                         string dtype,
                         int simdwidth,
                         bool storea,
                         bool storeb,
                         int maxLocalMemElems,
                         int padding,
                         string& kernel,
                         bool useImageA = false,
                         bool useImageB = false){
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
	bool isDouble = (dtype.compare("double")==0) ? true:false; 
	if(isDouble && simdwidth!=2 && (useImageA || useImageB)) return false;
	if(!isDouble && simdwidth!=4 && (useImageA || useImageB)) return false;
    //const int unroll = ktile;
    if(storea){
        /*Number of rows of A per workgroup = ktile. Has to be divisble by lsizey. */
        if(ktile%lsizex!=0) return false;
        /*Number of columns of A per workgroup = htile*lsizex*/
        if((htile*lsizex)%(simdwidth*lsizey)!=0) return false;
    }else{
        if(htile%simdwidth!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = wtile*lsizey. Has to be divisble by lsizex*simdwidth */
        if(ktile%lsizex!=0) return false;
        if(((wtile*lsizey)%(lsizey*simdwidth))!=0) return false;
    }else{
        if(wtile%simdwidth!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;

    if(wtile%simdwidth!=0 || htile%simdwidth!=0) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += ktile*(htile*lsizex+padding);
    if(storeb) numLocalMemElems += ktile*(wtile*lsizey+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //if(ktile%unroll!=0) return false;
    //cout<<"Check 4 passed"<<endl;
	stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
	ss<<"const int htile ="<<htile<<";\n";
	ss<<"const int wtile ="<<wtile<<";\n";
	ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
	ss<<"const int lx = "<<lsizex<<";\n"; 
	ss<<"const int ly = "<<lsizey<<";\n";
	ss<<"const int i = get_global_id(1);"<<endl;
	ss<<"const int j = get_global_id(0);"<<endl;
    //ss<<"const int unroll = "<<unroll<<";\n";
	ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
	ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    ss<<"int k;"<<endl;
	if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(ktile)<<"]["<<((htile/simdwidth)*lsizex+padding)<<"];"<<endl;
	}
	if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
	}
	for(int x=0;x<htile;x++){
		for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";
		}
	}
	ss<<"for(k=0;k<K;k+=ktile){"<<endl;
	if(storea){
		ss<<"const int gstartxA = get_group_id(1)*(htile/simdwidth)*lx;\n";
		for(int y=0;y<(ktile/lsizex);y++){
			for(int x=0;x<((htile*lsizex)/(simdwidth*lsizey));x++){
				ss<<" ldsA["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
				if(useImageA){
					if(isDouble){
                        ss<<"as_double2(read_imagei(A,sampler,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx )))";
					}else{
                        ss<<"(read_imagef(A,sampler,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx )))";
					}
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
				}else{
					ss<<"A[(k+"<<y<<"*lx+lidx)*(lda/simdwidth)+ gstartxA + lidy + "<<x<<"*ly];\n";
				}
			}
		}
	}
    if(storeb){
		ss<<"const int gstartxB = get_group_id(0)*(wtile/simdwidth)*ly;\n";
		for(int y=0;y<(ktile/lsizex);y++){
			for(int x=0;x<((wtile*lsizey)/(simdwidth*lsizey));x++){
				ss<<" ldsB["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
				if(useImageB){
					if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(lidy + "<<x<<"*ly+gstartxB,k+"<<y<<"*lx +lidx)))";
					}else{
                        ss<<"(read_imagef(B,sampler,(int2)(lidy + "<<x<<"*ly+gstartxB,k+"<<y<<"*lx +lidx)))";
					}
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
				}else{
					ss<<"B[(k+"<<y<<"*lx+lidx)*(ldb/simdwidth)+ gstartxB + lidy + "<<x<<"*ly];\n";
				}
            }
        }
    }
	if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    //ss<<" for(kk=0;kk<ktile;kk+=unroll){\n";
	ss<<"const int kk = 0;\n";
    for(int y =0; y < ktile; y++){
        for (int x = 0; x < wtile/simdwidth; x++) {
            ss << "  const " << dtype << simdwidth << " b" << x << "_" << y << " = ";
            if(storeb){
                ss << "ldsB[kk+"<<y<<"][lidy*(wtile/simdwidth)+"<<x<<"];\n";
            }else{
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2( read_imagei(B,sampler,(int2)(j*(wtile/simdwidth)+"<<x<<",k+kk+"<<y<<")))";
                    }else{
                        ss<<"( read_imagef(B,sampler,(int2)(j*(wtile/simdwidth)+"<<x<<",k+kk+"<<y<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss << "B[(k+kk+"<<y<<")*(ldb/simdwidth)+j*(wtile/simdwidth)+"<<x<<"];\n";
                }
            }
        }
        for (int x = 0; x < htile/simdwidth; x++) {
            ss << "  const " << dtype << simdwidth << " a" << x << "_" << y << " = ";
            if(storea){
                ss << "ldsA[kk+"<<y<<"]["<<"lidx*(htile/simdwidth)+"<<x<<"];\n";
            }else{
                if(useImageA){
                    if(isDouble) {
                        ss<<"as_double2(read_imagei(A,sampler,(int2)(i*(htile/simdwidth)+"<<x<<",k+kk+"<<y<<")))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(i*(htile/simdwidth)+"<<x<<",k+kk+"<<y<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss << "A[(k+kk+"<<y<<")*(lda/simdwidth)+i*(htile/simdwidth)+"<<x<<"];"<<endl;
                }
            }
            for(int xoff=0;xoff<simdwidth;xoff++){
                int row = x*simdwidth + xoff;
                for(int w=0;w<wtile/simdwidth; w++){
                    if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
                    ss<<"  sum"<<row<<"_"<<w<<" += "<<"a"<<x<<"_"<<y;
                    if(simdwidth>1){
                        ss<<".s";
                        for(int m=0;m<simdwidth;m++) ss<<xoff;
                    }
                    ss<<"*b"<<w<<"_"<<y<<";\n";
                    if(isDouble){ss<<"#else"<<endl;
                        ss<<"  sum"<<row<<"_"<<w<<" = fma( "<<"a"<<x<<"_"<<y;
                        if(simdwidth>1){
                            ss<<".s";
                            for(int m=0;m<simdwidth;m++) ss<<xoff;
                        }
                        ss<<",b"<<w<<"_"<<y<<",sum"<<row<<"_"<<w<<");\n";
                        ss<<"#endif"<<endl;
                    }
                }
            }
        }
    }
    //ss<<" }\n";

	if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
	ss<<"}"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss << "+= alpha*sum"<<i<<"_"<<(j/simdwidth);
                if(simdwidth>1) ss<<".s"<<(j%simdwidth);
                //ss<<" + ";
                //ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                //ss<<"C[(i*"<<htile<<"+ i)*N + j*"<<wtile<<"+"<<offset<<"]";
                ss << ";" << endl;
        }
    }
	ss<<"}"<<endl;
	kernel = ss.str();
	return true;
}

bool genKernelNNOff(int lsizex,
                         int lsizey,
                         int htile,
                         int wtile,
                         int ktile,
                         string dtype,
                         int simdwidth,
                         bool storea,
                         bool storeb,
                         int maxLocalMemElems,
                         int padding,
                         string& kernel,bool useImageA,bool useImageB){
    bool isDouble = (dtype.compare("double")==0);
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
    if(storea){
        /*Number of rows of A per workgroup = htile*lsizex. Has to be divisble by lsizex. Trivially satisfied */
        /*Number of columns of A per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(simdwidth*lsizey)!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = wtile*lsizey. Has to be divisble by lsizey*simdwidth */
        if(ktile%lsizex!=0) return false;
        if(((wtile*lsizey)%(lsizey*simdwidth))!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;

    if(wtile%simdwidth!=0 || ktile%simdwidth!=0) return false;
	if(!storea && !storeb && ktile>simdwidth) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += htile*lsizex*(ktile+padding);
    if(storeb) numLocalMemElems += ktile*(wtile*lsizey+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //cout<<"Check 4 passed"<<endl;
	stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
	ss<<"const int htile ="<<htile<<";\n";
	ss<<"const int wtile ="<<wtile<<";\n";
	ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
	ss<<"const int lx = "<<lsizex<<";\n"; 
	ss<<"const int ly = "<<lsizey<<";\n";
	ss<<"const int i = get_global_id(1);"<<endl;
	ss<<"const int j = get_global_id(0);"<<endl;
	ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
	ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    ss<<"int k;"<<endl;
	if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
	}
	if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
	}
	for(int x=0;x<htile;x++){
		for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";

		}
	}
    ss<<"for(k=0;k<K;k+=ktile){"<<endl;
	if(storea){
        ss<<"const int gstartxA = get_group_id(1)*htile*lx;\n";
        for(int x=0;x<htile;x++){
            for(int y=0;y<(ktile/(lsizey*simdwidth));y++){
                ss<<" ldsA["<<x<<"*lx + lidx]["<<y<<"*ly + lidy] = ";
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k/simdwidth +"<<y<<"*ly+lidy,gstartxA+"<<x<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k/simdwidth+"<<y<<"*ly+lidy,gstartxA+"<<x<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(gstartxA+"<<x<<"*lx + lidx)*(lda/simdwidth) + k/simdwidth + "<<y<<"*ly + lidy];\n";
                }
			}
		}
	}
    if(storeb){
        ss<<"const int gstartxB = get_group_id(0)*(wtile/simdwidth)*ly;\n";
        for(int x=0;x<(wtile/simdwidth);x++){
            for(int y=0;y<(ktile/lsizex);y++){
                ss<<" ldsB["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(gstartxB + lidy + "<<x<<"*ly,k+"<<y<<"*lx +lidx)))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(gstartxB + lidy + "<<x<<"*ly,k+"<<y<<"*lx +lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(k+"<<y<<"*lx+lidx)*(ldb/simdwidth)+ gstartxB + lidy + "<<x<<"*ly];\n";
                }
            }
        }
    }
	if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    /*for (int x = 0; x < wtile/simdwidth; x++) {
        for (int y = 0; y < ktile; y++) {
            ss << "  const " << dtype << simdwidth << " b" << x << "_" << y << " = ";
            if(storeb){
                ss << "ldsB["<<y<<"][lidy*(wtile/simdwidth)+"<<x<<"];\n";
            }else{
                ss << "B[(k+"<<y<<")*(ldb/simdwidth)+j*(wtile/simdwidth)+"<<x<<"];\n";
            }
        }
    }*/
    for (int k = 0; k < ktile/simdwidth; k++) {
		for (int koff = 0; koff < simdwidth; koff++) {
			for (int x = 0; x < wtile/simdwidth; x++) {
				int rowB = k*simdwidth+koff;
				ss << "  const " << dtype << simdwidth << " b" << x << "_" << rowB << " = ";
				if(storeb){
                    ss << "ldsB["<<rowB<<"][lidy+ly*"<<x<<"];\n";
				}else{
                    if(useImageB){
                        if(isDouble){
                            ss<<"as_double2(read_imagei(B,sampler,(int2)(get_group_id(0)*(wtile/simdwidth)*ly + lidy+"<<x<<"*ly,k+"<<rowB<<")))";
                        }else{
                            ss<<"(read_imagef(B,sampler,(int2)(get_group_id(0)*(wtile/simdwidth)*ly + lidy+"<<x<<"*ly,k+"<<rowB<<")))";
                        }
                        ss<<".s";
                        for(int s=0;s<simdwidth;s++) ss<<s;
                        ss<<";\n";

                    }else{
                        ss << "B[(k+"<<rowB<<")*(ldb/simdwidth)+get_group_id(0)*(wtile/simdwidth)*ly + lidy+"<<x<<"*ly];\n";
                    }

				}
			}
		}
        for(int y =0; y < htile; y++){
            ss << "  const " << dtype << simdwidth << " a" << k << "_" << y << " = ";
            if(storea){
                ss << "ldsA["<<y<<"*lx+lidx]["<<k<<"];\n";
            }else{
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k/simdwidth+"<<k<<",get_group_id(1)*htile*lx + lx*"<<y<<"+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k/simdwidth+"<<k<<",get_group_id(1)*htile*lx + lx*"<<y<<"+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                   ss << "A[(get_group_id(1)*htile*lx + lx*"<<y<<"+lidx)*(lda/simdwidth) + k/simdwidth+"<<k<<"];"<<endl;
                }

            }
            for(int koff=0;koff<simdwidth;koff++){
				int rowB = (k*simdwidth+koff);
				for(int x = 0;x<(wtile/simdwidth);x++){
                    if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
                    ss<<"sum"<<y<<"_"<<x<<" +=a"<<k<<"_"<<y;
                    if(simdwidth>1){
                        ss<<".s";
                        for(int t=0;t<simdwidth;t++) ss<<koff;
                    }
                    ss<<"*b"<<x<<"_"<<rowB<<";\n";
                    if(isDouble){ ss<<"#else"<<endl;
                        ss<<"sum"<<y<<"_"<<x<<" = fma(a"<<k<<"_"<<y;
                        if(simdwidth>1){
                            ss<<".s";
                            for(int t=0;t<simdwidth;t++) ss<<koff;
                        }
                        ss<<",b"<<x<<"_"<<rowB<<",";
                        ss<<"sum"<<y<<"_"<<x<<");\n";
                        ss<<"#endif"<<endl;
                    }
				}
            }
        }
    }

	if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
	ss<<"}"<<endl;
    ss<<"const unsigned int Cx = get_group_id(1)*htile*lx;"<<endl;
    ss<<"const unsigned int Cy = get_group_id(0)*wtile*ly;"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[( Cx+ lidx+lx*"<< i << ")*ldc + Cy + simdwidth*lidy + simdwidth*ly*" << j/simdwidth << "+"<<(j%simdwidth)<<"]";
                ss << "= alpha*sum"<<i<<"_"<<(j/simdwidth);
                if(simdwidth>1) ss<<".s"<<(j%simdwidth);
                ss<<" + beta*";
                ss << "C[( Cx+ lidx+lx*"<< i << ")*ldc + Cy + simdwidth*lidy + simdwidth*ly*" << j/simdwidth << "+"<<(j%simdwidth)<<"]";
                //ss<<"C[(i*"<<htile<<"+ i)*N + j*"<<wtile<<"+"<<offset<<"]";
                ss << ";" << endl;
        }
    }
	ss<<"}"<<endl;
	kernel = ss.str();
	return true;
}

static bool genKernelNNCons(int lsizex,
                         int lsizey,
                         int htile,
                         int wtile,
                         int ktile,
                         string dtype,
                         int simdwidth,
                         bool storea,
                         bool storeb,
                         int maxLocalMemElems,
                         int padding,
                         string& kernel,bool useImageA,bool useImageB){
    bool isDouble = (dtype.compare("double")==0);
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
    if(storea){
        /*Number of rows of A per workgroup = htile*lsizex. Has to be divisble by lsizex. Trivially satisfied */
        /*Number of columns of A per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(simdwidth*lsizey)!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = wtile*lsizey. Has to be divisble by lsizey*simdwidth */
        if(ktile%lsizex!=0) return false;
        if(((wtile*lsizey)%(lsizey*simdwidth))!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;

    if(wtile%simdwidth!=0 || ktile%simdwidth!=0) return false;
    if(!storea && !storeb && ktile>simdwidth) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += htile*lsizex*(ktile+padding);
    if(storeb) numLocalMemElems += ktile*(wtile*lsizey+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //cout<<"Check 4 passed"<<endl;
    stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
    ss<<"const int htile ="<<htile<<";\n";
    ss<<"const int wtile ="<<wtile<<";\n";
    ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
    ss<<"const int lx = "<<lsizex<<";\n";
    ss<<"const int ly = "<<lsizey<<";\n";
    ss<<"const int i = get_global_id(1);"<<endl;
    ss<<"const int j = get_global_id(0);"<<endl;
    ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
    ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    ss<<"int k;"<<endl;
    if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";

        }
    }
    ss<<"for(k=0;k<K;k+=ktile){"<<endl;
    if(storea){
        ss<<"const int gstartxA = get_group_id(1)*htile*lx;\n";
        for(int x=0;x<htile;x++){
            for(int y=0;y<(ktile/(lsizey*simdwidth));y++){
                ss<<" ldsA["<<x<<"*lx + lidx]["<<y<<"*ly + lidy] = ";
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k/simdwidth +"<<y<<"*ly+lidy,gstartxA+"<<x<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k/simdwidth+"<<y<<"*ly+lidy,gstartxA+"<<x<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(gstartxA+"<<x<<"*lx + lidx)*(lda/simdwidth) + k/simdwidth + "<<y<<"*ly + lidy];\n";
                }
            }
        }
    }
    if(storeb){
        ss<<"const int gstartxB = get_group_id(0)*(wtile/simdwidth)*ly;\n";
        for(int x=0;x<(wtile/simdwidth);x++){
            for(int y=0;y<(ktile/lsizex);y++){
                ss<<" ldsB["<<y<<"*lx + lidx]["<<x<<"*ly + lidy] = ";
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(gstartxB + lidy + "<<x<<"*ly,k+"<<y<<"*lx +lidx)))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(gstartxB + lidy + "<<x<<"*ly,k+"<<y<<"*lx +lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(k+"<<y<<"*lx+lidx)*(ldb/simdwidth)+ gstartxB + lidy + "<<x<<"*ly];\n";
                }
            }
        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    /*for (int x = 0; x < wtile/simdwidth; x++) {
        for (int y = 0; y < ktile; y++) {
            ss << "  const " << dtype << simdwidth << " b" << x << "_" << y << " = ";
            if(storeb){
                ss << "ldsB["<<y<<"][lidy*(wtile/simdwidth)+"<<x<<"];\n";
            }else{
                ss << "B[(k+"<<y<<")*(ldb/simdwidth)+j*(wtile/simdwidth)+"<<x<<"];\n";
            }
        }
    }*/
    for (int k = 0; k < ktile/simdwidth; k++) {
        for (int koff = 0; koff < simdwidth; koff++) {
            for (int x = 0; x < wtile/simdwidth; x++) {
                int rowB = k*simdwidth+koff;
                ss << "  const " << dtype << simdwidth << " b" << x << "_" << rowB << " = ";
                if(storeb){
                    ss << "ldsB["<<rowB<<"][lidy*(wtile/simdwidth)+"<<x<<"];\n";
                }else{
                    if(useImageB){
                        if(isDouble){
                            ss<<"as_double2(read_imagei(B,sampler,(int2)(j*(wtile/simdwidth) + "<<x<<",k+"<<rowB<<")))";
                        }else{
                            ss<<"(read_imagef(B,sampler,(int2)(j*(wtile/simdwidth) + "<<x<<",k+"<<rowB<<")))";
                        }
                        ss<<".s";
                        for(int s=0;s<simdwidth;s++) ss<<s;
                        ss<<";\n";

                    }else{
                        ss << "B[(k+"<<rowB<<")*(ldb/simdwidth)+j*(wtile/simdwidth)+"<<x<<"];\n";
                    }

                }
            }
        }
        for(int y =0; y < htile; y++){
            ss << "  const " << dtype << simdwidth << " a" << k << "_" << y << " = ";
            if(storea){
                ss << "ldsA["<<y<<"+lidx*htile]["<<k<<"];\n";
            }else{
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k/simdwidth +"<<k<<",i*htile+"<<y<<")))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k/simdwidth+"<<k<<",i*htile + "<<y<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";

                }else{
                    ss << "A[(i*htile  + "<<y<<")*(lda/simdwidth) + k/simdwidth+"<<k<<"];"<<endl;
                }
            }
            for(int koff=0;koff<simdwidth;koff++){
                int rowB = (k*simdwidth+koff);
                for(int x = 0;x<(wtile/simdwidth);x++){
                    if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
                    ss<<"sum"<<y<<"_"<<x<<" +=a"<<k<<"_"<<y;
                    if(simdwidth>1){
                        ss<<".s";
                        for(int t=0;t<simdwidth;t++) ss<<koff;
                    }
                    ss<<"*b"<<x<<"_"<<rowB<<";\n";
                    if(isDouble){ ss<<"#else"<<endl;
                        ss<<"sum"<<y<<"_"<<x<<" = fma(a"<<k<<"_"<<y;
                        if(simdwidth>1){
                            ss<<".s";
                            for(int t=0;t<simdwidth;t++) ss<<koff;
                        }
                        ss<<",b"<<x<<"_"<<rowB<<",";
                        ss<<"sum"<<y<<"_"<<x<<");\n";
                        ss<<"#endif"<<endl;
                    }
                }
            }
        }
    }

    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss << "= alpha*sum"<<i<<"_"<<(j/simdwidth);
                if(simdwidth>1) ss<<".s"<<(j%simdwidth);
                ss<<" + beta*";
                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                //ss<<"C[(i*"<<htile<<"+ i)*N + j*"<<wtile<<"+"<<offset<<"]";
                ss << ";" << endl;
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}


static bool genKernelNTOff(int lsizex, int lsizey, int htile, int wtile, int ktile, string dtype,int simdwidth, bool storea, bool storeb, int maxLocalMemElems,
                         int padding,string& kernel, bool useImageA,bool useImageB){
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
    if(storea){
        /*Number of rows of A per workgroup = htile*lsizex. Has to be divisble by lsizex. Trivially satisfied */
        /*Number of columns of A per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(simdwidth*lsizey)!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(lsizey*simdwidth)!=0) return false;
        /*Number of rows of B per workgroup = wtile*lsizey. Has to be divisble by lsizex*/
        if((wtile*lsizey)%lsizex!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;
    bool isDouble = (dtype.compare("double")==0);
    if(ktile%simdwidth!=0) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += htile*lsizex*(ktile+padding);
    if(storeb) numLocalMemElems += wtile*lsizey*(ktile+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //cout<<"Check 4 passed"<<endl;
    stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
    ss<<"const int htile ="<<htile<<";\n";
    ss<<"const int wtile ="<<wtile<<";\n";
    ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
    ss<<"const int lx = "<<lsizex<<";\n";
    ss<<"const int ly = "<<lsizey<<";\n";
    ss<<"const int i = get_global_id(1);\n";
    ss<<"const int j = get_global_id(0);\n";
    ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
    ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    ss<<"unsigned int k;"<<endl;
    if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(wtile*lsizey)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";
        }
    }
    ss<<"const unsigned int ldbs = ldb/simdwidth;\n";
    ss<<"const unsigned int ldas = lda/simdwidth;\n";
    ss<<"for(k=0;k<K/simdwidth;k+=(ktile/simdwidth)){"<<endl;
    if(storea){
        ss<<"const int gstartxA = get_group_id(1)*htile*lx;\n";
        for(int rowA=0;rowA<htile;rowA++){
            for(int colA=0;colA<(ktile/(lsizey*simdwidth));colA++){
                ss<<" ldsA["<<rowA<<"*lx + lidx]["<<colA<<"*ly + lidy] = ";
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k+"<<colA<<"*ly+lidy,gstartxA+"<<rowA<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k+"<<colA<<"*ly+lidy,gstartxA+"<<rowA<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(gstartxA+"<<rowA<<"*lx + lidx)*(lda/simdwidth) + k + "<<colA<<"*ly + lidy];\n";
                }
            }
        }
    }
    if(storeb){
        ss<<"const int gstartxB = get_group_id(0)*wtile*ly;\n";
        for(int rowB=0;rowB<(wtile*lsizey)/lsizex;rowB++){
            for(int colB=0;colB<(ktile/(lsizey*simdwidth));colB++){
                ss<<" ldsB["<<rowB<<"*lx + lidx]["<<colB<<"*ly + lidy] = ";
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(k+"<<colB<<"*ly+lidy,gstartxB+"<<rowB<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(k+"<<colB<<"*ly+lidy,gstartxB+"<<rowB<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(gstartxB+"<<rowB<<"*lx + lidx)*(ldb/simdwidth) + k + "<<colB<<"*ly + lidy];\n";
                }
            }
        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    for(int rowB =0;rowB<wtile;rowB++){
        for(int colB = 0;colB<ktile/simdwidth;colB++){
            ss<<" const "<<dtype<<simdwidth<<" b"<<rowB<<"_"<<colB<<" = ";
            if(storeb){
                ss<<"ldsB["<<rowB<<"*ly+lidy]["<<colB<<"];\n";
            }else{
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(k+"<<colB<<",get_group_id(0)*wtile*ly+"<<rowB<<"*ly+lidy)))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(k+"<<colB<<",get_group_id(0)*wtile*ly+"<<rowB<<"*ly+lidy)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(get_group_id(0)*wtile*ly + "<<rowB<<"*ly + lidy)*ldbs + k + "<<colB<<"];\n";
                }
            }
        }
    }
    for(int rowA=0;rowA<htile;rowA++){
        for(int colA =0;colA<ktile/simdwidth;colA++){
            ss<<" const "<<dtype<<simdwidth<<" a"<<rowA<<"_"<<colA<<" = ";
            if(storea){
                ss<<"ldsA["<<rowA<<"*lx+lidx]["<<colA<<"];\n";
            }else{
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k+"<<colA<<",get_group_id(1)*htile*lx+"<<rowA<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k+"<<colA<<",get_group_id(1)*htile*lx+"<<rowA<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(get_group_id(1)*htile*lx+ "<<rowA<<"*lx+lidx )*ldas +k + "<<colA<<"];\n";
                }
            }
            const int colB = colA;
            for(int rowB=0;rowB<wtile;rowB++){
                if(isDouble) ss<<"#ifndef FP_FAST_FMAF\n";
                ss<<" sum"<<rowA<<"_"<<rowB<<" += a"<<rowA<<"_"<<colA<<" * b"<<rowB<<"_"<<colB<<";\n";
                if(isDouble){
                    ss<<"#else"<<endl;
                    ss<<" sum"<<rowA<<"_"<<rowB<<" = fma(a"<<rowA<<"_"<<colA<<", b"<<rowB<<"_"<<colB<<", ";
                    ss<<" sum"<<rowA<<"_"<<rowB<<");\n";
                    ss<<"#endif"<<endl;
                }
            }

        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    ss<<"const unsigned int Cx = get_group_id(1)*htile*lx;"<<endl;
    ss<<"const unsigned int Cy = get_group_id(0)*wtile*ly;"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(Cx + " << i<< "*lx + lidx)*ldc + Cy + " << j << "*ly + lidy]";
                ss << "= alpha*(";
                for(int s=0;s<simdwidth;s++){
                    ss<<"sum"<<i<<"_"<<j;
                    if(simdwidth>1) ss<<".s"<<s;
                    if(s<(simdwidth-1)) ss<<"+";
                }
                ss<<")+ beta*(";
                ss << "C[(Cx + " << i<< "*lx + lidx)*ldc + Cy + " << j << "*ly + lidy]";
                ss<<");\n";
        }
    }
    //ss<<"}"<<endl;
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}




bool genKernelNTCons(int lsizex, int lsizey, int htile, int wtile, int ktile, string dtype,int simdwidth, bool storea, bool storeb, int maxLocalMemElems,
                         int padding,string& kernel, bool useImageA,bool useImageB){
    //cout<<"StoreA "<<storea<<" "<<"StoreB "<<storeb<<" "<<htile<<" "<<wtile<<" "<<" "<<ktile<<" "<<simdwidth<<" "<<lsizex<<" "<<lsizey<<" "<<unroll;
    //cout<<" "<<maxLocalMemElems<<" "<<endl;
    if(storea){
        /*Number of rows of A per workgroup = htile*lsizex. Has to be divisble by lsizex. Trivially satisfied */
        /*Number of columns of A per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(simdwidth*lsizey)!=0) return false;
    }

    if(storeb){
        /*Number of columns of B per workgroup = ktile. Has to be divisble by lsizey*simdwidth */
        if(ktile%(lsizey*simdwidth)!=0) return false;
        /*Number of rows of B per workgroup = wtile*lsizey. Has to be divisble by lsizex*/
        if((wtile*lsizey)%lsizex!=0) return false;
    }
    //cout<<"Check 2 passed"<<endl;
    bool isDouble = (dtype.compare("double")==0);
    if(ktile%simdwidth!=0) return false;
    int numLocalMemElems = 0;
    if(storea) numLocalMemElems += htile*lsizex*(ktile+padding);
    if(storeb) numLocalMemElems += wtile*lsizey*(ktile+padding);
    if(numLocalMemElems>maxLocalMemElems) return false;
    //cout<<"Check 3 passed"<<endl;
    //cout<<"Check 4 passed"<<endl;
    stringstream ss;
    ss<<"(";
    if(useImageA){
        ss<<"__read_only image2d_t A,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<simdwidth<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"*restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<" alpha,"<<dtype<<" beta){"<<endl;
    ss<<"const int htile ="<<htile<<";\n";
    ss<<"const int wtile ="<<wtile<<";\n";
    ss<<"const int ktile ="<<ktile<<";\n";
    ss<<"const int simdwidth="<<simdwidth<<";\n";
    ss<<"const int lx = "<<lsizex<<";\n";
    ss<<"const int ly = "<<lsizey<<";\n";
    ss<<"const int i = get_global_id(1);\n";
    ss<<"const int j = get_global_id(0);\n";
    ss<<"const unsigned int lidx = get_local_id(1);"<<endl;
    ss<<"const unsigned int lidy = get_local_id(0);"<<endl;
    ss<<"unsigned int k;"<<endl;
    if(storea){
        ss<<"__local "<<dtype<<simdwidth<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<simdwidth<<" ldsB["<<(wtile*lsizey)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile;y++){
            ss<<dtype<<simdwidth<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<simdwidth<<")(0);\n";
        }
    }
    ss<<"const unsigned int ldbs = ldb/simdwidth;\n";
    ss<<"const unsigned int ldas = lda/simdwidth;\n";
    ss<<"for(k=0;k<K/simdwidth;k+=(ktile/simdwidth)){"<<endl;
    if(storea){
        ss<<"const int gstartxA = get_group_id(1)*htile*lx;\n";
        for(int rowA=0;rowA<htile;rowA++){
            for(int colA=0;colA<(ktile/(lsizey*simdwidth));colA++){
                ss<<" ldsA["<<rowA<<"*lx + lidx]["<<colA<<"*ly + lidy] = ";
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k+"<<colA<<"*ly+lidy,gstartxA+"<<rowA<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k+"<<colA<<"*ly+lidy,gstartxA+"<<rowA<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(gstartxA+"<<rowA<<"*lx + lidx)*(lda/simdwidth) + k + "<<colA<<"*ly + lidy];\n";
                }
            }
        }
    }
    if(storeb){
        ss<<"const int gstartxB = get_group_id(0)*wtile*ly;\n";
        for(int rowB=0;rowB<(wtile*lsizey)/lsizex;rowB++){
            for(int colB=0;colB<(ktile/(lsizey*simdwidth));colB++){
                ss<<" ldsB["<<rowB<<"*lx + lidx]["<<colB<<"*ly + lidy] = ";
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(k+"<<colB<<"*ly+lidy,gstartxB+"<<rowB<<"*lx+lidx)))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(k+"<<colB<<"*ly+lidy,gstartxB+"<<rowB<<"*lx+lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(gstartxB+"<<rowB<<"*lx + lidx)*(ldb/simdwidth) + k + "<<colB<<"*ly + lidy];\n";
                }
            }
        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    for(int rowB =0;rowB<wtile;rowB++){
        for(int colB = 0;colB<ktile/simdwidth;colB++){
            ss<<" const "<<dtype<<simdwidth<<" b"<<rowB<<"_"<<colB<<" = ";
            if(storeb){
                ss<<"ldsB["<<rowB<<"+lidy*wtile]["<<colB<<"];\n";
            }else{
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2(read_imagei(B,sampler,(int2)(k+"<<colB<<",j*wtile+"<<rowB<<")))";
                    }else{
                        ss<<"(read_imagef(B,sampler,(int2)(k+"<<colB<<",j*wtile+"<<rowB<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(j*wtile+"<<rowB<<")*ldbs + k + "<<colB<<"];\n";
                }
            }
        }
    }
    for(int rowA=0;rowA<htile;rowA++){
        for(int colA =0;colA<ktile/simdwidth;colA++){
            ss<<" const "<<dtype<<simdwidth<<" a"<<rowA<<"_"<<colA<<" = ";
            if(storea){
                ss<<"ldsA["<<rowA<<"+lidx*htile]["<<colA<<"];\n";
            }else{
                if(useImageA){
                    if(isDouble){
                         ss<<"as_double2(read_imagei(A,sampler,(int2)(k+"<<colA<<",i*htile+"<<rowA<<")))";
                    }else{
                        ss<<"(read_imagef(A,sampler,(int2)(k+"<<colA<<",i*htile+"<<rowA<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(i*htile+"<<rowA<<")*ldas +k + "<<colA<<"];\n";
                }
            }
            const int colB = colA;
            for(int rowB=0;rowB<wtile;rowB++){
                if(isDouble) ss<<"#ifndef FP_FAST_FMAF\n";
                ss<<" sum"<<rowA<<"_"<<rowB<<" += a"<<rowA<<"_"<<colA<<" * b"<<rowB<<"_"<<colB<<";\n";
                if(isDouble){
                    ss<<"#else"<<endl;
                    ss<<" sum"<<rowA<<"_"<<rowB<<" = fma(a"<<rowA<<"_"<<colA<<", b"<<rowB<<"_"<<colB<<", ";
                    ss<<" sum"<<rowA<<"_"<<rowB<<");\n";
                    ss<<"#endif"<<endl;
                }
            }

        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss << "= alpha*(";
                for(int s=0;s<simdwidth;s++){
                    ss<<"sum"<<i<<"_"<<j;
                    if(simdwidth>1) ss<<".s"<<s;
                    if(s<(simdwidth-1)) ss<<"+";
                }
                ss<<")+ beta*(";
                 ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss<<");\n";
        }
    }
    //ss<<"}"<<endl;
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}


static string readFile(ifstream &fin){
	string data;
	if(!fin.is_open()) return "";
	while(!fin.eof()){
		string line;
		getline(fin,line);
		data += line;
	}
	return data;
}



class GemmDouble{
	public:
		typedef double realtype;
		typedef cl_double realtypecl;
		static bool isDouble(){return true;}
		static string name(){return "double";}
		static string gemmName(){return "dgemm";}
};

class GemmSingle{
	public:
		typedef float realtype;
		typedef cl_float realtypecl;
		static bool isDouble(){return false;}
		static string name(){return "float";}
		static string gemmName(){return "sgemm";}
};


template <typename T>
double testGemm(unsigned int N,cl_device_id dvc,cl_context ctx,cl_kernel krnl, RaijinGemmOptKernel& optkernel,
                RaijinTranspose *transObj,
                RaijinCopy *copyObj,
                RaijinScale *scaleObj,
                bool verify=true,int cpuPart=8){
	typedef typename T::realtype realtype;
	typedef typename T::realtypecl realtypecl;
	size_t size = sizeof(realtype) * N * N;
	cl_mem bufA, bufB, bufC;
	realtype *ptrA = new realtype[N * N];
	realtype *ptrB = new realtype[N * N];
	realtype *ptrC = new realtype[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
            if(optkernel.transA){
                ptrA[i * N + j] = 0.002 * j;
            }else{
                ptrA[i*N + j] = 0.002*i;
            }
        
            if(optkernel.transB){
                ptrB[i * N + j] = 0.002 * i;
            }else{
                ptrB[i*N + j] = 0.002*j;
            }

            ptrC[i * N + j] = 0;
		}
	}
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,NULL);
	cl_int errcode;
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
	const int niters = 3;
	double tdiff = 0;
	for(int i=0;i<niters;i++){
		RTimer rt;
		rt.start();
        cl_event evt = raijinApplyOpt<realtype>(krnl,optkernel,ctx,dvc,RaijinCL::RaijinRowMajor,optkernel.transA,optkernel.transB,N,N,N,1,
                                                bufA,N,bufB,N,0,bufC,N,params,transObj,copyObj,scaleObj);
		clWaitForEvents(1,&evt);
		rt.stop();
		if(i>0) tdiff += rt.getDiff();
	}
	tdiff /= (niters-1);
	if(verify){
	    clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, size, ptrC, 0, NULL, NULL);
		double totalerror = 0.0;
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				realtype calc = ptrC[i*N+j];
                realtype expected = N*(0.002*i)*(0.002*j);
				realtype val = calc - expected;
				if(val<0) val = -val;
                /*if(val>0.1){
                    cout<<kname<<" "<<i<<" "<<j<<" "<< val << " "<< calc << " "<< expected << endl;
                    exit(1);
                }*/
				totalerror += val;
			}
		}
		cout<<"Avg absolute error "<<(totalerror/(N*N))<<endl;
	}
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	delete[] ptrA;
	delete[] ptrB;
	delete[] ptrC;
	clReleaseCommandQueue(q);
	return 2.0e-9*N*(1.0*N)*(1.0*N)/tdiff;
}



template <typename T>
void tuneGemmCache(cl_context ctx, cl_device_id dvc,RaijinGemmOptKernel *optparams,unsigned int N,double *gflopbest){
	cout<<"Inside tuneGemmCache"<<endl;
	cout<<"Tuning "<<T::gemmName()<<endl;
	cl_int errcode;
	cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,&errcode);
    RaijinTranspose *transObj = new RaijinTranspose(dvc,ctx);
    RaijinCopy *copyObj = new RaijinCopy(ctx,dvc);
    RaijinScale *scaleObj = new RaijinScale(ctx,dvc);
	if(errcode!=CL_SUCCESS) cout<<"Error creating queue"<<endl;
	typedef typename T::realtype realtype;
	size_t size = sizeof(realtype)*N*N;
	int htiles[] = {2,2,4,4,8,4,8,16,4};
	int wtiles[] = {2,4,2,4,4,8,8,4,16};
    int ktiles[] = {1,2,4,8,16,32};
	int simdwidths[] = {1,2,4,8};
	int lsizesX[] = {4,8,4,16,16};
	int lsizesY[] = {8,8,16,4,16};
	int unrolls[] = {1,2,4,8};
    bool storeA[] = {true, false};
    bool storeB[] = {true, false};
	bool useImageA[] = {true,false};
	bool useImageB[] = {true,false};
	bool initialized = false;
	//double tbest = 0.0;
	string prgmbest;
	*gflopbest = 0.0;
    cl_device_type dvctype;
    cl_ulong lmemSize;
	cl_device_local_mem_type ltype;
    clGetDeviceInfo(dvc,CL_DEVICE_TYPE,sizeof(dvctype),&dvctype,NULL);
    clGetDeviceInfo(dvc,CL_DEVICE_LOCAL_MEM_SIZE,sizeof(lmemSize),&lmemSize,NULL);
	clGetDeviceInfo(dvc,CL_DEVICE_LOCAL_MEM_TYPE,sizeof(ltype),&ltype,NULL);
    cl_uint vecWidth;
    if(T::isDouble()) clGetDeviceInfo(dvc,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,sizeof(cl_uint),&vecWidth,NULL);
    else clGetDeviceInfo(dvc,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,sizeof(cl_uint),&vecWidth,NULL);
	bool imgA[] = {true,false};
	bool imgB[] = {true,false};
    for (int i = 5; i < 7; i++) {
        for (int j = 1; j < 4; j++) {
            for (int simdidx = 0; simdidx < 3;simdidx++) {
                for (int ktileidx = 0; ktileidx < 5; ktileidx++) {
                    for(int sa = 0 ; sa<2; sa++){
                        for(int sb = 0; sb <2 ; sb++){
                            for(int imgAidx=0;imgAidx<2;imgAidx++){
                                for(int imgBidx=0;imgBidx<2;imgBidx++){



									int ktile = ktiles[ktileidx];

									const int unr = ktile;
									//cout<<s<<" "<<bfidx<<" "<<splits[s]<<" "<<bfirsts[bfidx]<<endl;
									bool isAggregate = false;
									bool storec = false;
									int htile = htiles[i];
									int wtile = wtiles[i];
									bool useImageA = imgA[imgAidx];
									bool useImageB = imgB[imgBidx];
                                    bool transA = true;
                                    bool transB = false;
                                    if(dvctype!=CL_DEVICE_TYPE_GPU && (useImageA || useImageB)) continue;
                                    if(ltype!=CL_LOCAL && (storeA[sa] || storeB[sb])) continue;

									string body;
									const int simd = simdwidths[simdidx];
                                    //if(dvctype==CL_DEVICE_TYPE_CPU && simd!=vecWidth) continue;
                                    if(dvctype==CL_DEVICE_TYPE_GPU){
                                        if(T::isDouble() && simd>2) continue;
                                        else if(!(T::isDouble()) && simd>4) continue;
                                    }
									//int regest = (htile * wtile + htile * simd * u + wtile * simd * u);

									string dtype = T::name();
									int lx, ly;
									lx = lsizesX[j];
									ly = lsizesY[j];
									//clGetDeviceInfo(dvc,CL_KERNEL_PREFERRED_WORK_GROUP_MULTIPLE,
                                    unsigned int nVecRegs = htile*wtile;
                                    nVecRegs += (htile>wtile) ? (wtile/simd) : (htile/simd);
                                    //if(dvctype==CL_DEVICE_TYPE_CPU && nVecRegs>16) continue;
                                    bool kernSuc = genKernelTNOff(lx,ly,htile, wtile, ktile,dtype, simd, storeA[sa],storeB[sb],lmemSize/(sizeof(realtype))
                                        ,1,body,useImageA,useImageB);

                                    /*unsigned int nVecRegs = htile*wtile/simd;
                                    nVecRegs += (htile>wtile) ? (wtile/simd) : (htile/simd);
                                    if(dvctype==CL_DEVICE_TYPE_CPU && nVecRegs>16) continue;
                                    bool kernSuc = genKernelTNCons(lx,ly,htile, wtile, ktile,dtype, simd, storeA[sa],storeB[sb],lmemSize/(sizeof(realtype))
                                                                   ,1,body,useImageA,useImageB);*/

									if(!kernSuc) continue;


									//cout<<body<<endl;
									stringstream kernelstream;
									stringstream namestream;
									namestream << T::gemmName() << i << "_" << j << "_" << simdidx << "_" << ktileidx << "_" << sa << "_" <<sb<<"_"<<imgAidx<<"_"<<imgBidx;
									string kname = namestream.str();
									if(T::isDouble()) kernelstream<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
									if(simd==1) kernelstream<<"typedef "<<dtype<<" "<<dtype<<"1;"<<endl;
									//kernelstream << "__attribute((reqd_work_group_size(" << lsizesY[j] << "," << lsizesX[j] << ",1)))";
                                    if(useImageA || useImageB){
                                        kernelstream<<"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;"<<endl;
                                        kernelstream<<"float4 myread_imagef(__read_only image2d_t img,int2 pos){ return read_imagef(img,sampler,pos);\n}"<<endl;
                                    }
									kernelstream<<"__kernel ";
									if(isAggregate){
										kernelstream<<"__attribute__((reqd_work_group_size(1,1,1))) "<<endl;
									}else{
										kernelstream<<"__attribute__((reqd_work_group_size("<<ly<<","<<lx<<",1))) "<<endl;
									}
									kernelstream << "void " << kname;
									kernelstream << body;
									string kernelsrc = kernelstream.str();
									string klogname = kname +".cl";
									ofstream klog(klogname.c_str());
									klog<<kernelsrc<<endl;
									klog.close();
									const size_t len = kernelsrc.length();
									cl_int errcode1, errcode2;
									RTimer rt1, rt2, rt3;
									rt1.start();
									const char *srcbuf = kernelsrc.c_str();
									cl_program prg = clCreateProgramWithSource(ctx, 1, &srcbuf, (const size_t*) &len, &errcode1);
									cl_int bldcode = clBuildProgram(prg, 1, &dvc, "", NULL, NULL);
									cl_kernel krnl = clCreateKernel(prg, kname.c_str(), &errcode2);
									rt1.stop();
									cout<<"Compile time "<<rt1.getDiff()<<endl;
									if (errcode1 != CL_SUCCESS || errcode2 != CL_SUCCESS || bldcode != CL_SUCCESS) {
										/*cl::Program prgmcpp(prg);
										const cl::Device dvccpp(dvc);
										string buildlog = prgmcpp.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dvccpp);
										cout<<buildlog<<endl;*/
										size_t retbytes;
										cout << "Error creating program from source " << errcode1 << " " << errcode2 << " " << bldcode << endl;
										clGetProgramBuildInfo(prg, dvc, CL_PROGRAM_BUILD_LOG, 0, NULL, &retbytes);
										char *buildlog = new char[retbytes+1];
										clGetProgramBuildInfo(prg,dvc,CL_PROGRAM_BUILD_LOG,retbytes,buildlog,NULL);
										cout << "Buildlog " << retbytes<<" "<<buildlog << endl;
										//cout << "Error creating program from source " << errcode1 << " " << errcode2 << " " << bldcode << endl;
										cout << kernelsrc << endl;
										exit(-1);
										continue;

									} else {
										//string fname = kname+".cl";
										//ofstream of(fname.c_str());
										//of<<kernelsrc<<endl;
										//of.close();
										//cout<<"Time taken to compile "<<rt1.getDiff()<<endl;
										RaijinGemmOptKernel candidate;
										candidate.transA = transA;
										candidate.transB = transB;
										candidate.simdwidth = simd;
										candidate.htile = htile;
										candidate.wtile = wtile;
										candidate.ktile = ktile;
										candidate.lsizex = lx;
										candidate.lsizey = ly;
										candidate.kernel = kernelsrc;
										candidate.kname = kname;
										candidate.imageA = useImageA;
										candidate.imageB = useImageB;

										double gflops;
                                        size_t tuneSize = (dvctype==CL_DEVICE_TYPE_CPU)? 1024 : 2048;
                                        gflops = testGemm<T>(tuneSize, dvc, ctx, krnl,candidate,transObj,copyObj,scaleObj,true);

										clReleaseKernel(krnl);
										clReleaseProgram(prg);
										double bwidth = (htile+wtile)*gflops*sizeof(realtype)/(2*htile*wtile);
										cout<<"htile "<<htile<<" wtile "<<wtile<<" ktile "<<(ktile);
										cout<<" lx "<<lx<<" ly "<<ly<<" simd "<<simd<<" storeA? "<<storeA[sa]<<" storeB? "<<storeB[sb];
										cout<<" ImageA? "<<useImageA<<" ImageB? "<<useImageB<<endl;

										if (!initialized || (gflops > (*gflopbest)) && (gflops < 2500)) {
											*optparams = candidate;
											*gflopbest = gflops;
											initialized = true;
										}
										cout << "Gflops " << gflops << " Bwidth "<< bwidth<<" Best So Far "<<(*gflopbest)<<" "<<kname<<endl;

									}
								}
							}
						}
					}
				}
			}
		}
	}
	clReleaseCommandQueue(q);
    delete scaleObj;
    delete transObj;
    delete copyObj;

}


template <typename T>
bool tuneGemm(cl_platform_id platform, cl_device_id dvc,RaijinGemmOptKernel *optkernel,unsigned int N=1024){
	cout<<"Inside tuneGemm"<<endl;
	cl_context_properties conprop[3];
	conprop[0] = CL_CONTEXT_PLATFORM;
	conprop[1] = (cl_context_properties)platform;
	conprop[2] = (cl_context_properties)0;
	cl_int errcode;
	cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    if(errcode==CL_SUCCESS){
        double gflopbest=0.0;
        tuneGemmCache<T>(ctx,dvc,optkernel,N,&gflopbest);
    }else{
        cout<<"Could not successfully create context for this device"<<endl;
        return false;
    }
    clReleaseContext(ctx);
    return true;
}


void RaijinCL::raijinTuneSgemm(cl_device_id dvc){
    cl_platform_id platform;
    clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
    string dpath = raijinGetProfileFileName(dvc,"sgemm");
    RaijinGemmOptKernel sgemm;
    tuneGemm<GemmSingle>(platform,dvc,&sgemm);
    ofstream ofile(dpath.c_str());
    cout<<"File is good? "<<ofile.good()<<" Is Open? "<<ofile.is_open()<<endl;
    ofile<<sgemm<<endl;
    ofile.close();
}

void RaijinCL::raijinTuneDgemm(cl_device_id dvc){
    cl_platform_id platform;
    clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
    string dpath = raijinGetProfileFileName(dvc,"dgemm");
    RaijinGemmOptKernel dgemm;
    tuneGemm<GemmDouble>(platform,dvc,&dgemm);
    ofstream ofile(dpath.c_str());
    cout<<"File is good? "<<ofile.good()<<" Is Open? "<<ofile.is_open()<<endl;
    ofile<<dgemm<<endl;
    ofile.close();
}
