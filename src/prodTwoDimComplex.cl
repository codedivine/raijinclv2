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
__kernel void prodTwoDimComplex(__global double2 *input, int ioffset, __global double2 *output, int ooffset, int stride1,int stride2, int len,int lenDivGroupSize,int lenModGroupSize){
    size_t idx = get_group_id(0);
    size_t lidx = get_local_id(0);
    int start = idx*stride1 + lidx*lenDivGroupSize*stride2+ioffset;
    const int wgsize = 16;
    __local double2 ltemp[wgsize];
    int i;
    double2 threadSum = {1,0};
    for(i=0;i<lenDivGroupSize;i++){
        double2 temp = input[start + i*stride2];
        threadSum.x = threadSum.x*temp.x - threadSum.y*temp.y;
        threadSum.y = threadSum.x*temp.y + threadSum.y*temp.x;
    }
    ltemp[lidx] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lidx==0){
        double2 finalSum = {1,0};
        for(i=0;i<wgsize;i++){
            double2 temp = ltemp[i];
            finalSum.x = finalSum.x*temp.x - finalSum.y*temp.y;
            finalSum.y = finalSum.x*temp.y + finalSum.y*temp.x;
        }
        start = start + wgsize*lenDivGroupSize*stride2;
        for(i=0;i<lenModGroupSize;i++){
            double2 temp = input[start  i*stride2];
            finalSum.x = finalSum.x*temp.x - finalSum.y*temp.y;
            finalSum.y = finalSum.x*temp.y + finalSum.y*temp.x;
        }
        output[ooffset + idx] = finalSum;
    }
}
