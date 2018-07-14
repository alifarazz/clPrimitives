
/* filterWidth *must* be odd int */
__kernel void blurConvFilter(__read_only image2d_t inImg,
                             __write_only image2d_t outImg, int rows, int cols,
                             __constant float *filter, int filterWidth,
                             sampler_t sampler) {
  int X = get_global_id(0), Y = get_global_id(1);
  int halfWidth = filterWidth / 2;
  int4 sum = {0,0,0,0};
  int2 coord;
  int filterIdx = 0;

  for (int i = -halfWidth; i <= halfWidth; i++) {
    coord.y = Y + i;
    for (int j = -halfWidth; j <= halfWidth; j++) {
      coord.x = X + j;

      int4 pixel = read_imagei(inImg, sampler, coord);
      float intensity = filter[filterIdx];
      sum.x += pixel.x * intensity;
      sum.y += pixel.y * intensity;
      sum.z += pixel.z * intensity;
      filterIdx++;
    }
  }

  write_imagei(outImg, (int2){X, Y}, sum);
}
