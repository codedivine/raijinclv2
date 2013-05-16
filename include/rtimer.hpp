#ifndef __RTIMER_HPP_
#define __RTIMER_HPP_
#ifndef _WIN32
extern "C"{
#include <ctime>
#include <limits.h>
#include <stdlib.h>
}
class RTimer{
	struct timespec tv1,tv2;
	public:
	void start(){ clock_gettime(CLOCK_MONOTONIC,&tv1);}
	void stop(){ clock_gettime(CLOCK_MONOTONIC,&tv2);}
	double getDiff(){ return (tv2.tv_sec - tv1.tv_sec) + (1.0e-9)*(tv2.tv_nsec - tv1.tv_nsec);}
};
#endif

#ifdef _WIN32
#include <Windows.h>

class RTimer{
	LARGE_INTEGER freq;
	LARGE_INTEGER t1,t2;
	public:
	RTimer(){ QueryPerformanceFrequency(&freq);}
	void start(){ QueryPerformanceCounter(&t1);}
	void stop() { QueryPerformanceCounter(&t2);}
	double getDiff(){return ((t2.QuadPart-t1.QuadPart)*1.0)/freq.QuadPart;}
};
#endif
#endif
