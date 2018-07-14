__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

__kernel void rrotate(__read_only image2d_t inputImage,
                      __write_only image2d_t outputImage, int imageWidth,
                      int imageHieght, float theta) {
  /* Get global id for output coordiantes */
  int x = get_global_id(0), y = get_global_id(1);

  /* Compute image center */
  float x0 = imageWidth / 2, y0 = imageHieght / 2;

  /* Compute work-item's location relative to image's center */
  int xprime = x - x0, yprime = y - y0;

  float sinTheta = sin(theta), cosTheta = cos(theta);

  /* Compute *input* location */
  float2 readCoord;
  readCoord.x = xprime * cosTheta - yprime * sinTheta + x0;
  readCoord.y = xprime * sinTheta + yprime * cosTheta + y0;

  /* Read the input image */
  float3 c = read_imagef(inputImage, sampler, readCoord).xyz;

  /* Write the ouput image */
  write_imagef(outputImage, (int2)(x, y), (float4)(c.x, c.y, c.z, 0.0f));
}
