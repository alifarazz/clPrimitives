#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef OPENCL_1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#endif

// #include "./h/ocl.err.h"
// #include "./h/ocl.query.h"

#define RANDOM_NUMBER_MAX 1000

static inline auto ReadTextFile(const char *s)
{
  std::ifstream mfile(s);
  std::string content((std::istreambuf_iterator<char>(mfile)),
                      (std::istreambuf_iterator<char>()));
  mfile.close();
  return content;
}

static inline auto rand_gen()
{
  std::random_device rd;
  std::mt19937 mt(rd());
  return mt;
}

void
transpose_Host(float in[], float out[], int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      out[i * n +j] = in[j * n + i];
}

int main(int argc, char* argv[])
{
  using namespace std;

  ios_base::sync_with_stdio(false);

  if (argc != 4) {
    cerr << R"(Correct way to execute this program is:
./transpose platform-num data_size workgroup_size
For example: ./transpose 0 10000 512)";
    return 1;
  }

  size_t plat_num = atoi(argv[1]);
  int n_elem = atoi(argv[2]);
  int local_work_size = atoi(argv[3]);

  vector<float> h_data(n_elem * n_elem); // host, data
  vector<float> h_output(n_elem * n_elem); // host, output
  vector<float> d1_output(n_elem * n_elem); // device, output
  vector<float> d2_output(n_elem * n_elem); // device, output
  vector<float> d3_output(n_elem * n_elem); // device, output

  // fill A with random doubles
  uniform_real_distribution<> dis(0, RANDOM_NUMBER_MAX);
  // generate(h_data.begin(), h_data.end(), bind(dis, rand_gen()));
  generate(h_data.begin(), h_data.end(), [](){return rand() % RANDOM_NUMBER_MAX;});

  // copy(A.begin(), A.end(), ostream_iterator<double>(cout, " "));  // print all elements
  auto t1_serial = chrono::high_resolution_clock::now();
  transpose_Host(h_data.data(), h_output.data(), n_elem);
  auto t2_serial = chrono::high_resolution_clock::now();


  // Initialize OpenCL
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  assert(plat_num < platforms.size());
  auto platform = platforms[plat_num]; // here you can select between Intel, AMD or Nvidia
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  auto device = devices.front(); // here you can select between different Accelerators
  auto context = cl::Context(device);

  // Read the program source
  cl::Program program = cl::Program(context, ReadTextFile("./kernel.cl"), CL_FALSE);
  try {
#ifndef OPENCL_1
    program.build("-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math");
#else
    cout << "[WARN] operating in OpenCL 1.x mode\n";
    program.build("-cl-mad-enable  -cl-fast-relaxed-math");
#endif
  } catch (...) {
#ifndef OPENCL_1
    // Print build info for all devices
    cl_int buildErr = CL_SUCCESS;
    auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo) {
      cerr << pair.second << endl << endl;
    }
#else
    cerr << "building program failed" << endl;
#endif
    return 1;
  }

  // Select kernel
  auto ker_trans_per_elem = cl::Kernel(program, "transpose_parallel_per_element");
  // auto ker = cl::Kernel(program, "transpose_parallel_per_element");
  auto ker_trans_per_elem_tiled = cl::Kernel(program, "transpose_parallel_per_element_tiled");

  // Cerate the device memory buffers
  auto buf_src = cl::Buffer
    (context, CL_MEM_READ_ONLY, h_data.size() * sizeof(h_data[0]));
  auto buf_target = cl::Buffer
    (context, CL_MEM_WRITE_ONLY, d1_output.size() * sizeof(d1_output[0]));

  auto queue = cl::CommandQueue(context, device);

  chrono::time_point<chrono::high_resolution_clock> ti_trans_per_elem, te_trans_per_elem;
  {  //
    ker_trans_per_elem.setArg(0, buf_src);
    ker_trans_per_elem.setArg(1, buf_target);

    ti_trans_per_elem = chrono::high_resolution_clock::now();

    cl::NDRange global(n_elem, n_elem);
    cl::NDRange local(local_work_size, local_work_size);

    queue.enqueueWriteBuffer
      (buf_src, CL_TRUE, 0, h_data.size() * sizeof(float), h_data.data());
    queue.enqueueNDRangeKernel
      (ker_trans_per_elem, cl::NullRange, global, local);
    queue.enqueueReadBuffer
      (buf_target, CL_TRUE, 0, d1_output.size() * sizeof(float), d1_output.data());
    queue.finish();
    te_trans_per_elem = chrono::high_resolution_clock::now();
  }

  chrono::time_point<chrono::high_resolution_clock> ti_trans_per_elem_tiled, te_trans_per_elem_tiled;
  {  //
    ker_trans_per_elem_tiled.setArg(0, buf_src);
    ker_trans_per_elem_tiled.setArg(1, buf_target);
    ker_trans_per_elem_tiled.setArg(2, local_work_size * local_work_size * sizeof(float), nullptr);

    ti_trans_per_elem_tiled = chrono::high_resolution_clock::now();

    cl::NDRange global(n_elem, n_elem);
    cl::NDRange local(local_work_size, local_work_size);

    queue.enqueueWriteBuffer
      (buf_src, CL_TRUE, 0, h_data.size() * sizeof(float), h_data.data());
    queue.enqueueNDRangeKernel
      (ker_trans_per_elem_tiled, cl::NullRange, global, local);
    queue.enqueueReadBuffer
      (buf_target, CL_TRUE, 0, d2_output.size() * sizeof(float), d2_output.data());
    queue.finish();
    te_trans_per_elem_tiled = chrono::high_resolution_clock::now();
  }

  double
    serial_time                = chrono::duration<double, milli>(t2_serial - t1_serial).count(),
    ocl_per_elem_time          = chrono::duration<double, milli>(te_trans_per_elem - ti_trans_per_elem).count(),
    ocl_per_elem_tiled_time    = chrono::duration<double, milli>(te_trans_per_elem_tiled - ti_trans_per_elem_tiled).count();
  // copy(h_data.begin(), h_data.end(), ostream_iterator<float>(cout , " ")); cout << endl;

  cout << fixed << showpoint;
  cout.precision(6);
  cout << "[INFO] Serial time:\t"   << serial_time << "ms\n\n";
  // copy(h_output.begin(), h_output.end(), ostream_iterator<float>(cout , " ")); cout << endl;

  cout << "Transpose_per_element: " << ocl_per_elem_time << "\n";
  cout << "Verifiying transpose_per_element... "
       // << ((const char* []){"Error","Ok"})[mismatch(h_output.begin(), h_output.end(), d1_output.begin()).first == h_output.end()]
       << "\n";
  // copy(d1_output.begin(), d1_output.end(), ostream_iterator<float>(cout , " ")); cout << endl;

  cout << "Transpose_per_element_tiled: " << ocl_per_elem_tiled_time << "\n";
  cout << "Verifiying transpose_per_element_tiled... "
       // << ((const char* []){"Error","Ok"})[mismatch(h_output.begin(), h_output.end(), d2_output.begin()).first == h_output.end()]
       << "\n";
  // copy(d2_output.begin(), d2_output.end(), ostream_iterator<float>(cout , " ")); cout << endl;
  cout << endl;


  return 0;
}
