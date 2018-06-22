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
  int number_split = 1<<5;
  int w = 0, s = 0;
  float p[n_steps][number_split];
  int p_particles = n_particles;
  float pos[n_particles];
  int idx;

  for(int i=0; i<n_particles; i++) {
    pos[i] = 0.0f;
  }

  while(true) {
    idx = 0;
    if(partition_func(&p_particles, n_steps, number_split) == true) {
      while(w < n_steps) {
        int errorcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, number_split, p[w], -1.0f, 1.0f);
        if(errorcode != VSL_STATUS_OK) {
          printf("Error code %d", errorcode);
        }
        w += 1;
      }
      #pragma omp parallel for
      for (int j = 0; j < n_steps; j++) {
        //Intel MKL function to generate random numbers
        #pragma omp simd
        for(int k = 0; k < number_split; k++) {
          pos[k + idx] = dist_func(pos[idx], alpha, p[j][k]);
        }
      }
      idx += number_split;
    } else {
      break;
    }
  }

  #pragma omp simd reduction(+: n_escaped)
  for(int i = 0; i < n_particles; i++) {
    if (pos[i] > x_threshold) n_escaped++;
  }
  return n_escaped;
}