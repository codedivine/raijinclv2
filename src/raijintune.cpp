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
#include "raijin.hpp"
#include "raijin_complex.hpp"
#include <CL/cl.h>
#define RAIJIN_MAX_PLATFORMS 20
#define RAIJIN_MAX_DEVICE_PER_PLATFORM 10
using namespace std;
using namespace RaijinCL;


int main(int argc, char**argv){
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
				cout<<"Supports fp64? "<<supportsFp64(dvc)<<" Is amd_64? "<<isAmd64(dvc)<<endl;
    		}
    	}

    }else if(argv[1][0]=='D'){
    	int platid = atoi(argv[2]);
    	int devid = atoi(argv[3]);
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
        string kernelId(argv[4]);
        cout<<"Kernel ID "<<kernelId<<endl;
        if(kernelId.compare("sgemm")==0){
            raijinTuneSgemm(dvc);
        }else if(kernelId.compare("dgemm")==0){
            raijinTuneDgemm(dvc);
        }else if(kernelId.compare("zgemm")==0){
            raijinTuneZgemm(dvc);
        }else if(kernelId.compare("cgemm")==0){
            raijinTuneCgemm(dvc);
        }else if(kernelId.compare("strans")==0){
            RaijinTranspose::tuneStrans(dvc);
        }else if(kernelId.compare("dtrans")==0){
            RaijinTranspose::tuneDtrans(dvc);
        }else if(kernelId.compare("ctrans")==0){
            RaijinTranspose::tuneCtrans(dvc);
        }else if(kernelId.compare("ztrans")==0){
            RaijinTranspose::tuneZtrans(dvc);
        }

    }

}
