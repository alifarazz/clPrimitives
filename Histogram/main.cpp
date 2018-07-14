#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/gil/extension/io/jpeg_io.hpp>

#include "../utils.hpp"

namespace gil = boost::gil;

#define HIST_BINS 256

int main() {
  // auto beg = std::chrono::high_resolution_clock::now();

  gil::rgb8_image_t gilImage;
  gil::jpeg_read_image("../lena.small.jpg", gilImage);

  const int imageElements = gilImage.width() * gilImage.height();
  const size_t imageSize = imageElements * sizeof(int);
  const int histogramSize = HIST_BINS * sizeof(int);
  int *img = new int[imageElements];

  for (int x = 0, i = 0; x < gilImage.width(); ++x) {
    auto it = gilImage._view.col_begin(x);
    for (int y = 0; y < gilImage.height(); ++y)
      img[i++] = boost::gil::at_c<0>(it[y]);
  }
  // auto end = std::chrono::high_resolution_clock::now();
  /* std::cout << "time : "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(end -
     beg) .count()
	    << "milli sec" << std::endl; */

  int *hOutputHistogram = new int[histogramSize];

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
    cl::Buffer bufInputImage = cl::Buffer(context, CL_MEM_READ_ONLY, imageSize);
    cl::Buffer bufOutputHistogram =
	cl::Buffer(context, CL_MEM_WRITE_ONLY, imageSize);

    // Copy the input data to the input buffers using command-queue for first
    // deivce
    queue.enqueueWriteBuffer(bufInputImage, CL_TRUE, 0, imageSize, img);
    queue.enqueueFillBuffer(bufOutputHistogram, 0, 0, imageSize);

    // Read the program source
    std::string sourceCode = ReadTextFile("./histogram.cl");
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
    cl::Kernel vecadd_kernel(program, "histogram");

    // Set the kernel arguments
    vecadd_kernel.setArg(0, bufInputImage);
    vecadd_kernel.setArg(1, imageElements);
    vecadd_kernel.setArg(2, bufOutputHistogram);

    // Execute the kernel
    cl::NDRange global(imageElements);
    cl::NDRange local(256);
    queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, global, local);
    // Copy the output data back to the host
    queue.enqueueReadBuffer(bufOutputHistogram, CL_TRUE, 0, histogramSize,
			    hOutputHistogram);
    queue.flush();

    std::cout.flush();
  } catch (cl::Error error) {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }

  for (size_t i = 0; i < histogramSize / sizeof(int); i++)
    std::cout << i << ": " << hOutputHistogram[i] << '\n';

  delete[] hOutputHistogram;
  delete[] img;

  return 0;
}
