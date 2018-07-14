#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::string ReadTextFile(const char *s) {
  std::ifstream mfile(s);
  std::string content((std::istreambuf_iterator<char>(mfile)),
		      (std::istreambuf_iterator<char>()));
  mfile.close();
  return content;
}

int main() {
  const int elements = 2048 * 16;
  size_t datasize = elements * sizeof(int);

  int *A = new int[elements];
  int *B = new int[elements];
  int *C = new int[elements];

  for (int i = 0; i < elements; i++)
    A[i] = B[i] = i;

  try {
  // Query for platforms
  std::vector<cl::Platform> platform;
  cl::Platform::get(&platform);

  // Get a list of devices on this platform
  std::vector<cl::Device> devices;
  platform[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  // Create a context for the devices
  cl::Context context(devices);

  // Create a command-queue for the first device
  cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

  // Cerate the device memory buffers
  cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
  cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
  cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);

  // Copy the input data to the input buffers using command-queue for first
  // deivce
  queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
  queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);

  // Read the program source
  std::string sourceCode = ReadTextFile("./adder.cl");
  cl::Program::Sources source(
      1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

  // Create the program from the source code
  cl::Program program = cl::Program(context, source);

  // Build the program for the devices
  try {
    program.build(devices);
  } catch (cl::Error error) {
    cl::STRING_CLASS buildlog;
    program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildlog);
    std::cout << buildlog << std::endl;
  }

  // Create the kernel
  cl::Kernel vecadd_kernel(program, "vecadd");

  // Set the kernel arguments
  vecadd_kernel.setArg(0, bufferA);
  vecadd_kernel.setArg(1, bufferB);
  vecadd_kernel.setArg(2, bufferC);

  // Execute the kernel
  cl::NDRange global(elements);
  cl::NDRange local(256);
  queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, local);
  // Copy the output data back to the host
  queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);
  queue.flush();

  // for (size_t i = 0; i < elements; i++)
    // std::cout << C[i] << ' ' ;
  // std::cout << std::endl;
  } catch (cl::Error error) {
  std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
