#include "distribution.h"

//distribution function definition
#pragma omp declare simd
float dist_func(const float alpha, float rn) {
  return delta_max*sinf(alpha*rn)*expf(-rn*rn);
}