#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

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

#include <CL/cl2.hpp>

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


int main(int argc, char* argv[])
{
  using namespace std;

  ios_base::sync_with_stdio(false);

  if (argc != 5) {
    cerr << R"(Correct way to execute this program is:
./multi_cq platform-num data_size workgroup_size command_queue_count
For example: ./multi_cq 0 10000 512 4 )";
    return 1;
  }

  auto OPERATION = [](int32_t X){return sinf(X)/1319+cosf(X)/1317+cosf(X+13)*sinf(X-13);};

  int platform_num    {atoi(argv[1])},
      n_elem          {atoi(argv[2])},
      local_work_size {atoi(argv[3])},
      cq_count        {atoi(argv[4])}; // cq == command_queue

  vector<int32_t> h_data(n_elem); // host, data
  vector<int32_t> h_output(n_elem); // host, output
  vector<int32_t> d_output(n_elem); // device, output

  // fill A with random doubles
  uniform_int_distribution<> dis(0, RANDOM_NUMBER_MAX);
  generate(h_data.begin(), h_data.end(), bind(dis, rand_gen()));


  // copy(A.begin(), A.end(), ostream_iterator<double>(cout, " "));  // print all elements
  auto t1_serial = chrono::high_resolution_clock::now();
  transform(h_data.begin(), h_data.end(), h_output.begin(), OPERATION);  // cpp = GOD
  auto t2_serial = chrono::high_resolution_clock::now();


  // Initialize OpenCL
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  assert((size_t)platform_num < platforms.size());
  auto platform = platforms[platform_num]; // here you can select between Intel, AMD or Nvidia
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
    cout << "[WARN] Compling kernel in opencl 1.x mode\n";
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
  auto kernel = cl::Kernel(program, "vector_operation");

  // Cerate the device memory buffers
  auto buf_src = cl::Buffer
    (context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, h_data.size() * sizeof(h_data[0]));
  auto buf_target = cl::Buffer
    (context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, d_output.size() * sizeof(d_output[0]));

  // Set the kernel arguments
  kernel.setArg(0, buf_src);
  kernel.setArg(1, buf_target);

  vector<cl::CommandQueue> queues(cq_count);
  generate(queues.begin(), queues.end(), [&](){return cl::CommandQueue(context, device);}); // cpp = 2xGOD

  auto t1_ocl = chrono::high_resolution_clock::now();

  cl_int ocl_err {CL_SUCCESS};
  int cq_elem_count = n_elem / cq_count;
  int cq_elem_size = cq_elem_count * sizeof(h_data[0]);
  for (int i = 0; i < cq_count; ++i) {
    int offset = i * cq_elem_count;
    int offset_with_size = i * cq_elem_size;

    cl::NDRange global(cq_elem_count);
    cl::NDRange local(local_work_size);
    cl::NDRange offset_ndrange(offset);
    vector<cl::Event> ndrange_deps(1), read_deps(1);

    // 3rd(offset) param is on device, 4th(amount) and 5th(start void ptr) are on host
    ocl_err |= queues[i].enqueueWriteBuffer
      (buf_src, CL_FALSE, offset_with_size, cq_elem_size, h_data.data() + offset, nullptr, ndrange_deps.data());
    ocl_err |= queues[i].enqueueNDRangeKernel
      (kernel, offset_ndrange, global, local, &ndrange_deps, read_deps.data());
    ocl_err |= queues[i].enqueueReadBuffer
      (buf_target, CL_FALSE, offset_with_size, cq_elem_size, d_output.data() + offset, &read_deps, nullptr);
  }
  for_each(queues.begin(), queues.end(), [&](auto& q){ocl_err |= q.finish();});

  auto t2_ocl = chrono::high_resolution_clock::now();


  double serial_time = chrono::duration<double, milli>(t2_serial - t1_serial).count() * 10,
         ocl_time    = chrono::duration<double, milli>(t2_ocl - t1_ocl).count() * 10;
  cout << fixed << showpoint;
  cout.precision(6);
  cout << "[INFO] Serial time:\t"   << serial_time << "ms"
       << "\n[INFO] OpenCL time:\t" << ocl_time    << "ms" <<  '\n';

  auto location = mismatch(h_output.begin(), h_output.end(), d_output.begin());  // cpp = GOD
  // TODO: ooutput oclerr
  cout << "ocl_err status = " << ocl_err << '\n';
  if (location.first == h_output.end()) {
    cout.precision(2);
    cout << "[INFO] Test PASS! No mismatch found!\n" << "[INFO] Achived speedup of "
         << serial_time / ocl_time << "x" << endl;
  }
  else
    cout << "[FAIL]  There is a mismatch at location " << (location.first - h_output.begin())
         << "\n\twhere HOST contains " << *location.first << " and DEVICE contains "
         << *location.second << endl;

  return 0;
}
