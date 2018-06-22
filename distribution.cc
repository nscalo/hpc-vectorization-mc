#include "distribution.h"

//distribution function definition
#pragma omp declare simd
float dist_func(float buffer, const float alpha, float rn) {
  return buffer + delta_max*sinf(alpha*rn)*expf(-rn*rn);
}