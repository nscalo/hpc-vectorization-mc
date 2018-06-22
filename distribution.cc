#include "distribution.h"

//distribution function definition
#pragma omp declare simd
float dist_func(float buffer, const float alpha, float rn) {
  return buffer + delta_max*sinf(alpha*rn)*expf(-rn*rn);
}

#pragma omp declare simd
bool partition_func(int *status, int n_steps, int number_split) {
  status -= number_split;
  if(status > 0) return true;
  return false;
}