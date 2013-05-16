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
__kernel void prodOneDimComplex(__global double2 *input, int ioffset, __global double2 *output, int ooffset, int stride, int start, int lenModStripLen){
const int stripLen = 16;
int id = get_global_id(0);
int i;
double2 sum = {0,0};
if(id==0){
	for(i=0;i<lenModStripLen;i++){
		double2 temp = input[ioffset + stride*i];
		sum.x = temp.x*sum.x - temp.y*sum.y;
		sum.y = temp.x*sum.y + temp.y*sum.x;
	}
}else{
	for(i=0;i<stripLen;i+=4){
		int start = ioffset + stride*(lenModStripLen + (id-1)*stripLen);
		double2 temp = input[start + stride*(i+0)];
		sum.x = temp.x*sum.x - temp.y*sum.y;
		sum.y = temp.x*sum.y + temp.y*sum.x;
		temp = input[start + stride*(i+1)];
		sum.x = temp.x*sum.x - temp.y*sum.y;
		sum.y = temp.x*sum.y + temp.y*sum.x;
		temp = input[start + stride*(i+2)];
		sum.x = temp.x*sum.x - temp.y*sum.y;
		sum.y = temp.x*sum.y + temp.y*sum.x;
		temp = input[start + stride*(i+3)];
		sum.x = temp.x*sum.x - temp.y*sum.y;
		sum.y = temp.x*sum.y + temp.y*sum.x;

	}
}
output[ooffset+ id] = sum;
}
