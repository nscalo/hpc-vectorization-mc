#include <mkl.h>
#include "distribution.h"


//vectorize this function based on instruction on the lab page
// int diffusion(const int n_particles, 
//               const int n_steps, 
//               const float x_threshold,
//               const float alpha, 
//               VSLStreamStatePtr rnStream) {
//   int n_escaped=0;
//   #pragma omp parallel for
//   for (int i = 0; i < n_particles; i++) {
//     float x = 0.0f;
//     #pragma omp simd reduction(+: x)
//     for (int j = 0; j < n_steps; j++) {
//       float rn;
      
//       //Intel MKL function to generate random numbers
//       vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 1, &rn, -1.0, 1.0);
      
//       x += dist_func(alpha, rn); 
//     }
//     if (x > x_threshold) n_escaped++;
//   }
//   return n_escaped;
// }

#pragma omp simd
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  
  int n_escaped=0;
  // float rn[n_particles];
  // float x[n_particles];

  // for(int i=0; i<n_particles; i++) {
  //   x[i] = 0.0f;
  // }

  // for(int j=0; j<n_steps; j++) {
  //   float *p = rn;
  //   vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, p, -1.0, 1.0);

  //   for(int k=0; k<n_particles; k++) {
  //     x[k] = dist_func(x[k], alpha, rn[k]);
  //   }
  // }

  // #pragma omp simd reduction(+: n_escaped)
  // for(int k=0; k<n_particles; k++) {
  //   if(x[k] > x_threshold) n_escaped += 1;
  // }
 
  float rn;
  
  #pragma omp simd reduction(+: n_escaped)
  for(int i = 0; i < n_particles; i++) {
    float x = 0.0f;
    #pragma omp simd reduction(+: x)
    for (int j = i; j < i + n_steps; j++) {
      //Intel MKL function to generate random numbers
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 1, &rn, -1.0, 1.0);

      x = dist_func(x, alpha, rn);
    }
    if (x > x_threshold) n_escaped += 1;
  }
  return n_escaped;
}