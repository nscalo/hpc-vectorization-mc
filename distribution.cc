#include "distribution.h"

// global variables must not be modified
float get_delta_max() {
  return delta_max;
}

//distribution function definition
#pragma omp declare simd
float dist_func(const float alpha, float rn) {
  return get_delta_max()*sinf(alpha*rn)*expf(-rn*rn);
}