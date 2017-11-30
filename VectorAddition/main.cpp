#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>

int main()
{
  std::cout << "hello world" << std::endl;

}
