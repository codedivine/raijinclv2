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

#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "raijin_complex.hpp"
#include "rtimer.hpp"
#include <CL/cl.h>


using namespace std;
using namespace RaijinCL;

struct GemmCsingle{
    typedef cl_float2 ctype;
    typedef cl_float basetype;
    static bool isDouble(){return false;}
    static string gemmName(){return "cgemm";}
    static string name(){return "float";}
};

struct GemmCdouble{
    typedef cl_double2 ctype;
    typedef cl_double basetype;
    static bool isDouble(){return true;}
    static string gemmName(){return "zgemm";}
    static string name(){return "double";}
};

static bool genCkernelTNOff(int lsizex,
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

    if(isDouble && simdwidth>1 && (useImageA || useImageB)) return false;
    if(!isDouble && simdwidth>2 && (useImageA || useImageB)) return false;
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
        ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(ktile)<<"]["<<((htile/simdwidth)*lsizex+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";
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
                        ss<<"(myread_imagef(A,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                        ss<<"(myread_imagef(B,(int2)(lidy + "<<x<<"*ly+gstartxB,k+"<<y<<"*lx +lidx)))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
            ss << "  const " << dtype << (simdwidth*2) << " b" << x << "_" << y << " = ";
            if(storeb){
                ss << "ldsB[kk+"<<y<<"][ly*"<<x<<"+lidy];\n";
            }else{
                if(useImageB){
                    if(isDouble){
                        ss<<"as_double2( read_imagei(B,sampler,(int2)((get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy,k+kk+"<<y<<")))";
                    }else{
                        ss<<"( myread_imagef(B,(int2)((get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy,k+kk+"<<y<<")))";
                    }
                    ss<<".s";
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss << "B[(k+kk+"<<y<<")*(ldb/simdwidth)+ (get_group_id(0)*(wtile/simdwidth)+"<<x<<")*ly + lidy];\n";
                }
            }
        }
    }
     for (int y = 0; y < ktile; y++){
         for (int x = 0; x < htile/simdwidth; x++) {


             ss << "  const " << dtype << (simdwidth*2) << " a" << x << "_" << y << " = ";
             if(storea){
                 ss << "ldsA[kk+"<<y<<"][lx*"<<x<<"+lidx];\n";
             }else{
                 if(useImageA){
                     if(isDouble) {
                         ss<<"as_double2(read_imagei(A,sampler,(int2)((get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx,k+kk+"<<y<<")))";
                     }else{
                         ss<<"(myread_imagef(A,(int2)((get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx,k+kk+"<<y<<")))";
                     }
                     ss<<".s";
                     for(int s=0;s<simdwidth*2;s++) ss<<s;
                     ss<<";\n";
                 }else{
                     ss << "A[(k+kk+"<<y<<")*(lda/simdwidth)+ (get_group_id(1)*(htile/simdwidth)+"<<x<<")*lx + lidx];"<<endl;
                 }
             }
             //for(int x=0;x<htile/simdwidth;x++){
             for(int xoff=0;xoff<simdwidth;xoff++){
                 int row = x*simdwidth + xoff;
                 for(int w=0;w<wtile/simdwidth; w++){
                     ss<<"  sum"<<row<<"_"<<w;
                     ss<<" = fmaComplex"<<simdwidth;
                     ss<<"(a"<<x<<"_"<<y<<".s";
                     for(int m=0;m<simdwidth;m++){
                        ss<<(2*xoff);
                        ss<<(2*xoff+1);
                     }
                     ss<<",b"<<w<<"_"<<y<<",";
                     ss<<"  sum"<<row<<"_"<<w;
                     ss<<");\n";

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
                    ss << "= mulComplex1(alpha,sum"<<(i*simdwidth+ii)<<"_"<<j;
                    if(simdwidth>1){
                        ss<<".s"<<(2*jj)<<(2*jj+1);
                    }
                    ss<<") + mulComplex1(beta,";
                    ss << "C[( (get_group_id(1)*htile+"<<i<<"*simdwidth)*lx + lidx*simdwidth+ "<<ii<<")*ldc + (get_group_id(0)*wtile + " << j << "*simdwidth)*ly + lidy*simdwidth+" << jj << "]";
                    //ss << "C[(i*" << htile << "+ " << i << ")*ldc + (get_group_id(0)*wtile + " << j << "*simdwidth)*ly + lidy*simdwidth+" << jj << "]";
                    ss << ");" << endl;
                }
            }
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

static bool genCkernelNTCons(int lsizex, int lsizey, int htile, int wtile, int ktile, string dtype,int simdwidth, bool storea, bool storeb, int maxLocalMemElems,
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(wtile*lsizey)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";
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
            ss<<" const "<<dtype<<(simdwidth*2)<<" b"<<rowB<<"_"<<colB<<" = ";
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
            ss<<" const "<<dtype<<(simdwidth*2)<<" a"<<rowA<<"_"<<colA<<" = ";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(i*htile+"<<rowA<<")*ldas +k + "<<colA<<"];\n";
                }
            }
        }
        for(int colA =0;colA<ktile/simdwidth;colA++){
            const int colB = colA;
            for(int rowB=0;rowB<wtile;rowB++){
                ss<<" sum"<<rowA<<"_"<<rowB;
                ss<<" = fmaComplex"<<simdwidth<<"(a"<<rowA<<"_"<<colA<<",b"<<rowB<<"_"<<colB<<",";
                ss<<" sum"<<rowA<<"_"<<rowB;
                ss<<");\n";
            }

        }
    }
    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss << "= mulComplex1(alpha,(";
                for(int s=0;s<simdwidth;s++){
                    ss<<"sum"<<i<<"_"<<j<<".s";
                    ss<<(2*s)<<(2*s+1);
                    if(s<(simdwidth-1)) ss<<"+";
                }
                ss<<"))+ mulComplex1(beta,";
                 ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss<<");\n";
        }
    }
    //ss<<"}"<<endl;
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

static bool genCkernelTNCons(int lsizex,
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(ktile)<<"]["<<((htile/simdwidth)*lsizex+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";
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
                        ss<<"(read_imagef(A,sampler,(int2)(lidy+"<<x<<"*ly+gstartxA,k+"<<y<<"*lx + lidx)))";
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
            ss << "  const " << dtype << (simdwidth*2) << " b" << x << "_" << y << " = ";
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
            ss << "  const " << dtype << (simdwidth*2) << " a" << x << "_" << y << " = ";
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
                    ss<<"  sum"<<row<<"_"<<w;
                    ss<<" = fmaComplex"<<simdwidth<<"(a"<<x<<"_"<<y;
                    if(simdwidth>1){
                        ss<<".s";
                        for(int m=0;m<simdwidth;m++) {
                            ss<<(2*xoff)<<(2*xoff+1);
                        }
                    }
                    ss<<",b"<<w<<"_"<<y<<",";
                    ss<<"  sum"<<row<<"_"<<w<<");\n";
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
                ss << "= mulComplex1(alpha,sum"<<i<<"_"<<(j/simdwidth);
                if(simdwidth>1) ss<<".s"<<(2*(j%simdwidth))<<(2*(j%simdwidth)+1);
                ss<<") + ";
                ss << "mulComplex1(beta,C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                //ss<<"C[(i*"<<htile<<"+ i)*N + j*"<<wtile<<"+"<<offset<<"]";
                ss << ");" << endl;
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

static bool genCkernelNTOff(int lsizex, int lsizey, int htile, int wtile, int ktile, string dtype,int simdwidth, bool storea, bool storeb, int maxLocalMemElems,
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(wtile*lsizey)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
            ss<<" const "<<dtype<<(simdwidth*2)<<" b"<<rowB<<"_"<<colB<<" = ";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"B[(get_group_id(0)*wtile*ly + "<<rowB<<"*ly + lidy)*ldbs + k + "<<colB<<"];\n";
                }
            }
        }
    }
    for(int rowA=0;rowA<htile;rowA++){
        for(int colA =0;colA<ktile/simdwidth;colA++){
            ss<<" const "<<dtype<<(simdwidth*2)<<" a"<<rowA<<"_"<<colA<<" = ";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";
                }else{
                    ss<<"A[(get_group_id(1)*htile*lx+ "<<rowA<<"*lx+lidx )*ldas +k + "<<colA<<"];\n";
                }
            }
            const int colB = colA;
            for(int rowB=0;rowB<wtile;rowB++){
                ss<<" sum"<<rowA<<"_"<<rowB;
                ss<<" = fmaComplex"<<simdwidth<<"(a"<<rowA<<"_"<<colA<<",b"<<rowB<<"_"<<colB<<",";
                ss<<" sum"<<rowA<<"_"<<rowB;
                ss<<");\n";

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
                ss << "= mulComplex1(alpha,(";
                for(int s=0;s<simdwidth;s++){
                    ss<<"sum"<<i<<"_"<<j;
                    ss<<".s"<<(2*s)<<(2*s+1);
                    if(s<(simdwidth-1)) ss<<"+";
                }
                ss<<"))+ mulComplex1(beta,(";
                ss << "C[(Cx + " << i<< "*lx + lidx)*ldc + Cy + " << j << "*ly + lidy]";
                ss<<"));\n";
        }
    }
    //ss<<"}"<<endl;
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

static bool genCkernelNNCons(int lsizex,
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";

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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                ss << "  const " << dtype << (simdwidth*2) << " b" << x << "_" << rowB << " = ";
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
                        for(int s=0;s<simdwidth*2;s++) ss<<s;
                        ss<<";\n";

                    }else{
                        ss << "B[(k+"<<rowB<<")*(ldb/simdwidth)+j*(wtile/simdwidth)+"<<x<<"];\n";
                    }

                }
            }
        }
        for(int y =0; y < htile; y++){
            ss << "  const " << dtype << (simdwidth*2) << " a" << k << "_" << y << " = ";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";

                }else{
                    ss << "A[(i*htile  + "<<y<<")*(lda/simdwidth) + k/simdwidth+"<<k<<"];"<<endl;
                }
            }
            for(int koff=0;koff<simdwidth;koff++){
                int rowB = (k*simdwidth+koff);
                for(int x = 0;x<(wtile/simdwidth);x++){
                    ss<<"sum"<<y<<"_"<<x<<" = fmaComplex"<<simdwidth<<"(a"<<k<<"_"<<y;
                    ss<<".s";
                    for(int t=0;t<simdwidth;t++) ss<<(2*koff)<<(2*koff+1);
                    ss<<",b"<<x<<"_"<<rowB<<",";
                    ss<<"sum"<<y<<"_"<<x<<");\n";
                }
            }
        }
    }

    if(storea || storeb) ss<<" barrier(CLK_LOCAL_MEM_FENCE);"<<endl;
    ss<<"}"<<endl;
    for (int i = 0; i < htile; i++) {
        for (int j = 0; j < wtile; j++) {

                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "]";
                ss << "= mulComplex1(alpha,sum"<<i<<"_"<<(j/simdwidth);
                ss<<".s"<<2*(j%simdwidth)<<(2*(j%simdwidth)+1);
                ss<<") + mulComplex1(beta,";
                ss << "C[(i*" << htile << "+ " << i << ")*ldc + j*" << wtile << "+" << j << "])";
                ss << ";" << endl;
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

static bool genCkernelNNOff(int lsizex,
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
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict A,";
    }
    if(useImageB){
            ss<<"__read_only image2d_t B,";
    }else{
        ss<<"const __global "<<dtype<<(simdwidth*2)<<" *restrict B,";
    }
    ss<<"__global "<<dtype<<"2 *restrict C,unsigned int lda,unsigned int ldb,unsigned int ldc,unsigned int K,"<<dtype<<"2 alpha,"<<dtype<<"2 beta){"<<endl;
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
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsA["<<(htile*lsizex)<<"]["<<((ktile/simdwidth)+padding)<<"];"<<endl;
    }
    if(storeb){
        ss<<"__local "<<dtype<<(simdwidth*2)<<" ldsB["<<(ktile)<<"]["<<((wtile/simdwidth)*lsizey+padding)<<"];"<<endl;
    }
    for(int x=0;x<htile;x++){
        for(int y=0;y<wtile/simdwidth;y++){
            ss<<dtype<<(simdwidth*2)<<" sum"<<x<<"_"<<y<<" = ("<<dtype<<(simdwidth*2)<<")(0);\n";

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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
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
                ss << "  const " << dtype << (simdwidth*2) << " b" << x << "_" << rowB << " = ";
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
                        for(int s=0;s<simdwidth*2;s++) ss<<s;
                        ss<<";\n";

                    }else{
                        ss << "B[(k+"<<rowB<<")*(ldb/simdwidth)+get_group_id(0)*(wtile/simdwidth)*ly + lidy+"<<x<<"*ly];\n";
                    }

                }
            }
        }
        for(int y =0; y < htile; y++){
            ss << "  const " << dtype << (simdwidth*2) << " a" << k << "_" << y << " = ";
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
                    for(int s=0;s<simdwidth*2;s++) ss<<s;
                    ss<<";\n";
                }else{
                   ss << "A[(get_group_id(1)*htile*lx + lx*"<<y<<"+lidx)*(lda/simdwidth) + k/simdwidth+"<<k<<"];"<<endl;
                }

            }
            for(int koff=0;koff<simdwidth;koff++){
                int rowB = (k*simdwidth+koff);
                for(int x = 0;x<(wtile/simdwidth);x++){

                    ss<<"sum"<<y<<"_"<<x<<" = fmaComplex"<<simdwidth<<"(a"<<k<<"_"<<y;

                    ss<<".s";
                    for(int t=0;t<simdwidth;t++) ss<<(2*koff)<<(2*koff+1);

                    ss<<",b"<<x<<"_"<<rowB<<",";
                    ss<<"sum"<<y<<"_"<<x<<");\n";

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
                ss << "= mulComplex1(alpha,sum"<<i<<"_"<<(j/simdwidth);
                const int off = j%simdwidth;
                ss<<".s"<<(2*off)<<(2*off+1);
                ss<<") + mulComplex1(beta,";
                ss << "C[( Cx+ lidx+lx*"<< i << ")*ldc + Cy + simdwidth*lidy + simdwidth*ly*" << j/simdwidth << "+"<<(j%simdwidth)<<"])";
                //ss<<"C[(i*"<<htile<<"+ i)*N + j*"<<wtile<<"+"<<offset<<"]";
                ss << ";" << endl;
        }
    }
    ss<<"}"<<endl;
    kernel = ss.str();
    return true;
}

template <typename T>
static double testGemmComplex(unsigned int N,cl_device_id dvc,cl_context ctx,cl_kernel krnl, RaijinGemmOptKernel& optkernel, RaijinTranspose *transObj,
RaijinCopy *copyObj,
RaijinScale *scaleObj,bool verify=true){
    typedef typename T::ctype ctype;
    typedef typename T::basetype basetype;
    size_t size = sizeof(ctype) * N * N;
    cl_mem bufA, bufB, bufC;
    ctype *ptrA = new ctype[N * N];
    ctype *ptrB = new ctype[N * N];
    ctype *ptrC = new ctype[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(optkernel.transA){
                ptrA[i * N + j].s[0] = 0.002 * j;
                ptrA[i*N +j].s[1] = 1;
            }else{
                ptrA[i*N + j].s[0] = 0.002*i;
                ptrA[i*N+j].s[1] = 1;
            }

            if(optkernel.transB){
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
    cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,NULL);
    cl_int errcode;
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &errcode);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size, NULL, &errcode);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &errcode);
    clEnqueueWriteBuffer(q, bufA, CL_TRUE, 0, size, ptrA, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bufB, CL_TRUE, 0, size, ptrB, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bufC, CL_TRUE, 0, size, ptrC, 0, NULL, NULL);
    clFlush( q);
    const int niters = 3;
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
		RaijinCleaner *cleaner = new RaijinCleaner;
        cl_event evt = raijinApplyOpt<ctype>(q,cleaner,krnl,optkernel,ctx,dvc,RaijinCL::RaijinRowMajor,optkernel.transA,optkernel.transB,N,N,N,
                                                    alpha,bufA,N,bufB,N,beta,bufC,N,transObj,copyObj,scaleObj);
        clFinish(q);
		delete cleaner;
        rt.stop();
        if(i>0){
            tdiff += rt.getDiff();
            cout<<rt.getDiff()<<endl;
        }
    }
    tdiff /= (niters-1);
    if(verify){
        clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, size, ptrC, 0, NULL, NULL);
        double totalerror = 0.0;
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
        double avgerror = (totalerror)/(N*N);
        cout<<"Avg absolute error "<<(totalerror/(N*N))<<endl;
        //if(avgerror>1.0) exit(-1);
    }
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    delete[] ptrA;
    delete[] ptrB;
    delete[] ptrC;
    clReleaseCommandQueue(q);
    return 8.0e-9*N*(1.0*N)*(1.0*N)/tdiff;
}

string genCmulFuncs(bool isDouble){
    string dtype = (isDouble)? "double":"float";
    stringstream ss;

    //mulComplex1

	if(isDouble){
		ss<<"#ifdef cl_khr_fp64\n"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_khr_fp64 : enable"<<endl;
		ss<<"#else"<<endl;
		ss<<"#pragma OPENCL EXTENSION cl_amd_fp64 : enable"<<endl;
		ss<<"#endif"<<endl;
	}
    ss<<dtype<<"2 mulComplex1("<<dtype<<"2 a,"<<dtype<<"2 b){"<<endl;
    ss<<dtype<<"2 c;"<<endl;
    //if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
    //ss<<"c.x = a.x*b.x - a.y*b.y;"<<endl;
    //if(isDouble){
        //ss<<"#else"<<endl;
        ss<<dtype<<" temp = -a.y*b.y;"<<endl;
        ss<<"c.x = fma(a.x,b.x,temp);"<<endl;
        //ss<<"#endif"<<endl;
    //}
    //if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
    //ss<<"c.y = a.x*b.y + a.y*b.x;"<<endl;
    //if(isDouble){
        //ss<<"#else"<<endl;
        ss<<dtype<<" temp2 = a.y*b.x;"<<endl;
        ss<<"c.y = fma(a.x,b.y,temp2);"<<endl;
        //ss<<"#endif"<<endl;
    //}
    ss<<"return c;\n}"<<endl;

    //mulComplex2
    ss<<dtype<<"4 mulComplex2("<<dtype<<"4 a,"<<dtype<<"4 b){"<<endl;
    ss<<dtype<<"4 c;"<<endl;
    ss<<"c.s01 = mulComplex1(a.s01,b.s01); c.s23 = mulComplex1(a.s23,b.s23);"<<endl;
    ss<<"return c;\n}"<<endl;

    //fmaComplex1
    ss<<dtype<<"2 fmaComplex1("<<dtype<<"2 a,"<<dtype<<"2 b,"<<dtype<<"2 c){"<<endl;
    ss<<" "<<dtype<<"2 res;"<<endl;
    if(isDouble) ss<<"#ifndef FP_FAST_FMAF"<<endl;
    ss<<" res.x = a.x*b.x + c.x;"<<endl;
    ss<<" res.y = a.x*b.y + c.y;"<<endl;
    ss<<" res.x = -a.y*b.y + res.x;"<<endl;
    ss<<" res.y = a.y*b.x + res.y;"<<endl;
    if(isDouble){
        ss<<"#else"<<endl;
        ss<<" res.x = fma(-a.y,b.y,c.x);"<<endl;
		ss<<" res.y = fma(a.y,b.x,c.y);"<<endl;
        ss<<" res.x = fma(a.x,b.x,res.x);"<<endl;
        ss<<" res.y = fma(a.x,b.y,res.y);"<<endl;
        ss<<"#endif"<<endl;
    }
    ss<<" return res;"<<endl;
    ss<<"}"<<endl;

    //fmaComplex2
    ss<<dtype<<"4 fmaComplex2("<<dtype<<"4 a,"<<dtype<<"4 b,"<<dtype<<"4 c){"<<endl;
    ss<<dtype<<"4 res;"<<endl;
    ss<<"res.s01 = fmaComplex1(a.s01,b.s01,c.s01); res.s23 = fmaComplex1(a.s23,b.s23,c.s23);"<<endl;
    ss<<"return res;\n}"<<endl;

    /*ss<<dtype<<"2 are = a.s02,aim =a.s13;\n";
    ss<<dtype<<"2 bre = b.s02,bim= b.s13;\n";
    ss<<dtype<<"2 cre = c.s02,cim =c.s13;\n";
    ss<<dtype<<"2 rre = are*bre+cre; rre = -aim*bim+rre;\n";
    ss<<dtype<<"2 rim = are*bim+cim; rim = bre*aim+rim;\n";
    ss<<"rre = -aim*bim + rre;\n"<<endl;
    ss<<"rim = bre*aim + rim;\n";
    ss<<dtype<<"4 res; res.s02 = rre; res.s13 = rim;\n";
    ss<<"return res;\n}";*/


    return ss.str();
}


template <typename T>
static void tuneGemmComplex(cl_context ctx, cl_device_id dvc,RaijinGemmOptKernel *optparams,unsigned int N,double *gflopbest){
    cout<<"Inside tuneGemmCache"<<endl;
    cout<<"Tuning "<<T::gemmName()<<endl;
    cl_int errcode;
    cl_command_queue q = clCreateCommandQueue(ctx,dvc,0,&errcode);
    if(errcode!=CL_SUCCESS) cout<<"Error creating queue"<<endl;
    typedef typename T::ctype ctype;
    size_t size = sizeof(ctype)*N*N;
    int htiles[] = {2,4,4,8,4};
    int wtiles[] = {4,2,4,4,8};
    int ktiles[] = {1,2,4,8,16,32};
    int simdwidths[] = {1,2,4,8};
    int lsizesX[] = {4,8,8,4,16,16};
    int lsizesY[] = {8,4,8,16,4,16};
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
    RaijinTranspose transObj(dvc,ctx);
    RaijinCopy copyObj(ctx,dvc);
    RaijinScale scaleObj(ctx,dvc);
    cl_uint vecWidth;
    if(T::isDouble()) clGetDeviceInfo(dvc,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,sizeof(cl_uint),&vecWidth,NULL);
    else clGetDeviceInfo(dvc,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,sizeof(cl_uint),&vecWidth,NULL);
    bool imgA[] = {true,false};
    bool imgB[] = {true,false};
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int simdidx = 0; simdidx < 2;simdidx++) {
                for (int ktileidx = 0; ktileidx < 5; ktileidx++) {
                    for(int sa = 0 ; sa<1; sa++){
                        for(int sb = 0; sb <1 ; sb++){
                            for(int imgAidx=0;imgAidx<2;imgAidx++){
                                for(int imgBidx=0;imgBidx<2;imgBidx++){

                                    //if(T::isDouble() && simdidx>0) continue;

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
                                    int regest = 2*(htile * wtile + htile * simd + wtile * simd);
									if(regest>128) continue;

                                    string dtype = T::name();
                                    int lx, ly;
                                    lx = lsizesX[j];
                                    ly = lsizesY[j];

                                    unsigned int nVecRegs = htile*wtile;
                                    nVecRegs += (htile>wtile) ? (wtile/simd) : (htile/simd);
                                    //if(dvctype==CL_DEVICE_TYPE_CPU && nVecRegs>16) continue;
                                    bool kernSuc = genCkernelTNOff(lx,ly,htile, wtile, ktile,dtype, simd, storeA[sa],storeB[sb],lmemSize/(sizeof(ctype))
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
                                    kernelstream<<genCmulFuncs(T::isDouble())<<endl;
                                    if(useImageA || useImageB){
                                        kernelstream<<"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;"<<endl;
                                        kernelstream<<"float4 myread_imagef(__read_only image2d_t img,int2 pos){ return read_imagef(img,sampler,pos);\n}"<<endl;
                                    }
                                    kernelstream<<"__kernel ";
                                    if(isAggregate){
                                        kernelstream<<"__attribute__((reqd_work_group_size(1,1,1))) "<<endl;
                                    }else{
                                        kernelstream<<"__attribute__((reqd_work_group_size("<<ly<<","<<lx<<",1))) "<<endl;
                                    }
                                    kernelstream <<"void " << kname;
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
                                        size_t tuneSize = 2048;

                                        gflops = testGemmComplex<T>(tuneSize, dvc, ctx, krnl,candidate,&transObj,&copyObj,&scaleObj,true);

                                        clReleaseKernel(krnl);
                                        clReleaseProgram(prg);
                                        double bwidth = (htile+wtile)*gflops*sizeof(ctype)/(8*htile*wtile);
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

}

template <typename T>
bool tuneGemmComplex(cl_platform_id platform, cl_device_id dvc,RaijinGemmOptKernel *optkernel,unsigned int N=1024){
    cout<<"Inside tuneGemm"<<endl;
    cl_context_properties conprop[3];
    conprop[0] = CL_CONTEXT_PLATFORM;
    conprop[1] = (cl_context_properties)platform;
    conprop[2] = (cl_context_properties)0;
    cl_int errcode;
    cl_context ctx = clCreateContext(conprop,1,&dvc,NULL,NULL,&errcode);
    if(errcode==CL_SUCCESS){
        double gflopbest=0.0;
        tuneGemmComplex<T>(ctx,dvc,optkernel,N,&gflopbest);
    }else{
        cout<<"Could not successfully create context for this device"<<endl;
        return false;
    }
    clReleaseContext(ctx);
    return true;
}

void RaijinCL::raijinTuneZgemm(cl_device_id dvc){
    RaijinGemmOptKernel zgemmParams;
    cl_platform_id platform;
    clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
    string zpath = raijinGetProfileFileName(dvc,"zgemm");
    ofstream zfile(zpath.c_str());
    tuneGemmComplex<GemmCdouble>(platform,dvc,&zgemmParams);
    zfile<<zgemmParams<<endl;
    zfile.close();
}

void RaijinCL::raijinTuneCgemm(cl_device_id dvc){
    RaijinGemmOptKernel cgemmParams;
    cl_platform_id platform;
    clGetDeviceInfo(dvc,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform,NULL);
    string cpath = raijinGetProfileFileName(dvc,"cgemm");
    ofstream cfile(cpath.c_str());
    tuneGemmComplex<GemmCsingle>(platform,dvc,&cgemmParams);
    cfile<<cgemmParams<<endl;
    cfile.close();
}


