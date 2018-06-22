#include <mkl.h>
#include "distribution.h"
#include <cstdio>
#include <cstdlib>

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

int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  
  int n_escaped=0;
  int pow = 2;
  int p_particles = n_particles;
  p_particles /= pow;
  int shift_register = 6;
  int number_split = 1<<shift_register;
  int w = 0, s = 0;
  float rn[p_particles];
  
  float pos[n_particles];
  int idx;

  for(int i=0; i<n_particles; i++) {
    pos[i] = 0.0f;
  }

  
  while(s < n_steps) {
    idx = 0;
    int iter = 1;
    int i_idx = 0;
    while(iter <= 2) {
      int errorcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, p_particles, rn, -1.0f, 1.0f);
      if(errorcode != VSL_STATUS_OK) {
        printf("Error code %d", errorcode);
      }
      for(int idx = i_idx; idx < iter*p_particles; idx += number_split^pow) {
        int p_idx = 0;
        //Intel MKL function to generate random numbers
        for(int k = 0; k < number_split^(pow/2); k++) {
          for(int i = 0; i < number_split^(pow/2); i++) {
            pos[p_idx + idx] += dist_func(alpha, rn[p_idx + idx]);
            #pragma omp atomic
            p_idx += 1;
          }
        }
      }
      iter += 1;
      i_idx += p_particles;
    }
    s += 1;
  }

  #pragma omp simd reduction(+: n_escaped)
  for(int i = 0; i < n_particles; i++) {
    if (pos[i] > x_threshold) n_escaped++;
  }
  return n_escaped;
}