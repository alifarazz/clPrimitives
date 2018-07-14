#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <boost/gil/extension/io/jpeg_io.hpp>

#include "../utils.hpp"

namespace gil = boost::gil;
using namespace std;

static float gaussianBlurFilter[5][5] =
  {
   {
    1.0f / 273.0f,
    4.0f / 273.0f,
    7.0f / 273.0f,
    4.0f / 273.0f,
    1.0f / 273.0f
   },
   {
    4.0f / 273.0f,
    16.0f / 273.0f,
    26.0f / 273.0f,
    16.0f / 273.0f,
    4.0f / 273.0f
   },
   {
    7.0f / 273.0f,
    26.0f / 273.0f,
    41.0f / 273.0f,
    26.0f / 273.0f,
    7.0f / 273.0f,
   },
   {
    4.0f / 273.0f,
    16.0f / 273.0f,
    26.0f / 273.0f,
    16.0f / 273.0f,
    4.0f / 273.0f,
   },
   {
    1.0f / 273.0f,
    4.0f / 273.0f,
    7.0f / 273.0f,
    4.0f / 273.0f,
    1.0f / 273.0f
   }};

static const int gaussianBlurFilterWidth = 5;
static const size_t filterSize =
    gaussianBlurFilterWidth * gaussianBlurFilterWidth * sizeof(float);

int main() {
  gil::rgb8_image_t gilImage;
  gil::jpeg_read_image("../lena.small.jpg", gilImage);

  const int imgCols = gilImage.width(), imgRows = gilImage.height();
  const int imageElements = imgCols * imgRows;
  // imageElements * 4 because rgba
  int *hImg = readImage(gilImage, new int[imageElements * 4]); // utils

  gil::rgb8_image_t outGilImg(imgCols, imgRows);

  try {
    // Query for platforms
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // Create a context for the devices
    cl::Context context(devices[0]);

    // Create a command-queue for the first device
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

    // Cerate the device memory buffers
    cl::Image2D bufInputImage(context, CL_MEM_READ_ONLY,
			      cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32),
			      imgCols, imgRows, 0, hImg);
    cl::Image2D bufOutputImage(context, CL_MEM_WRITE_ONLY,
			       cl::ImageFormat(CL_RGBA, CL_SIGNED_INT32),
			       imgCols, imgRows);
    cl::Buffer bufFilter(context, CL_MEM_READ_ONLY, filterSize);

    // Offset within the image to copy
    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    // Region of image we want to pass
    cl::size_t<3> region;
    region[0] = (size_t)imgCols;
    region[1] = (size_t)imgRows;
    region[2] = 1;
    queue.enqueueWriteImage(bufInputImage, CL_TRUE, origin, region, 0, 0, hImg);
    queue.enqueueWriteBuffer(bufFilter, CL_TRUE, 0, filterSize,
			     gaussianBlurFilter);
    // queue.enqueueFillImage(bufOutputImage, cl_int4({0,0,0,0}), origin,
    // region);

    // Read the program source
    std::string sourceCode = ReadTextFile("./blur.cl");
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
    cl::Kernel rotate_kernel(program, "blurConvFilter");

    // Set the kernel arguments
    rotate_kernel.setArg(0, bufInputImage);
    rotate_kernel.setArg(1, bufOutputImage);
    rotate_kernel.setArg(2, (int)imgRows);
    rotate_kernel.setArg(3, (int)imgCols);
    rotate_kernel.setArg(4, bufFilter);
    rotate_kernel.setArg(5, gaussianBlurFilterWidth);
    rotate_kernel.setArg(6, cl::Sampler(context, CL_FALSE,
					CL_ADDRESS_CLAMP_TO_EDGE,
					CL_FILTER_NEAREST));

    // Execute the kernel
    cl::NDRange global(imgCols, imgRows); // for each pixel
    cl::NDRange local(4, 4);
    queue.enqueueNDRangeKernel(rotate_kernel, cl::NullRange, global, local);

    // Copy the output image back to the host
    queue.enqueueReadImage(bufOutputImage, CL_TRUE, origin, region, 0, 0, hImg);
    queue.flush();

  } catch (cl::Error error) {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    goto clean_exit;
  }

  writeImage(outGilImg, hImg); // utils

  gil::jpeg_write_view("./result.jpg", const_view(outGilImg)); // write image

clean_exit:
  delete[] hImg;

  return 0;
}
