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
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void sumMultiDimComplex(__global double2 *input, int ioffset, __global double2 *output, int ooffset, int ndims, int axis, __global const int *strides, __global const int *dims,
int len, int lenDivGroupSize, int lenModGroupSize){
    size_t groupidx = get_group_id(0);
    size_t lidx = get_local_id(0);
    int oidx = ooffset;
    int start = ioffset;
    const int wgsize = 16;
    __local double2 ltemp[wgsize];
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
    double2 threadSum = {0,0};
    for(i=0;i<lenDivGroupSize;i++){
        double2 temp = input[start + i*axisstride];
        threadSum.x += temp.x;
        threadSum.y += temp.y;
    }
    ltemp[lidx] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx==0){
        double2 finalSum = {0,0};
        for(i=0;i<wgsize;i++){
            double2 temp = ltemp[i];
            finalSum.x += temp.x;
            finalSum.y += temp.y;
        }
        start = start + wgsize*lenDivGroupSize*axisstride;
        for(i=0;i<lenModGroupSize;i++){
            double2 temp = input[start  i*axisstride];
            finalSum.x += temp.x;
            finalSum.y += temp.y;
        }
        output[ooffset + oidx] = finalSum;
    }
}
