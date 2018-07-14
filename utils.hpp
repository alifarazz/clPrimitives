#ifndef UTILS_H
#define UTILS_H

#include <boost/gil/extension/io/jpeg_io.hpp>
#include <fstream>
#include <iostream>

inline int *readImage(const boost::gil::rgb8_image_t &gilImage, int *buf) {
  int i = 0;

  for (int x = 0; x < gilImage.width(); ++x) {
    auto it = gilImage._view.col_begin(x);
    for (int y = 0; y < gilImage.height(); ++y) {
      buf[i++] = boost::gil::at_c<0>(it[y]);
      buf[i++] = boost::gil::at_c<1>(it[y]);
      buf[i++] = boost::gil::at_c<2>(it[y]);
      buf[i++] = 0; // boost::gil::at_c<3>(it[y]);
    }
  }
  return buf;
}

inline void writeImage(boost::gil::rgb8_image_t &gilImage, int *buf) {
  int i = 0;
  for (int x = 0; x < gilImage.width(); ++x) {
    auto it = gilImage._view.col_begin(x);
    for (int y = 0; y < gilImage.height(); ++y) {
      boost::gil::at_c<0>(it[y]) = buf[i++];
      boost::gil::at_c<1>(it[y]) = buf[i++];
      boost::gil::at_c<2>(it[y]) = buf[i++];
      i++;
      // boost::gil::at_c<3>(it[y]) = 0;i++;
    }
  }
}

inline std::string ReadTextFile(const char *s) {

  std::ifstream mfile(s);
  std::string content((std::istreambuf_iterator<char>(mfile)),
		      (std::istreambuf_iterator<char>()));
  mfile.close();
  return content;
}

#endif
