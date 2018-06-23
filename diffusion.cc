#include <mkl.h>
#include "distribution.h"
#include <cstdio>
#include <cstdlib>

//vectorize this function based on instruction on the lab page
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  
  int n_escaped=0;
  int z_number = 0;
 
  float pos[n_particles];
  int pow = 4;
  unsigned int nn_particles = n_particles * 1/pow;

  #pragma omp parallel for
  for(int i=0; i < n_particles; i++) {
    pos[i] = 0.0f;
  }

  int s = 0;
  while(s < n_steps) {
    // choose nn_particles first
    unsigned int i = 0;
    short int a = 1;
    // iterate through nn_particles
    iterate_particles(alpha, i, a, n_particles, rnStream, nn_particles, pos);
    iterate_particles(alpha, i, 2*a, n_particles, rnStream, nn_particles, pos);
    iterate_particles(alpha, i, 3*a, n_particles, rnStream, nn_particles, pos);
    iterate_particles(alpha, i, 4*a, n_particles, rnStream, nn_particles, pos);
    s += 1;
  }
  #pragma omp simd reduction(+: n_escaped)
  for(int i=0; i < n_particles; i++) {
    if (pos[i] > x_threshold) {
      n_escaped += 1;
    }
  }
  return n_escaped;
}

int iterate_particles(const float alpha, unsigned int i, short int a, int total_length, VSLStreamStatePtr rnStream, int nn_particles, float *pos) {
  
  float rn[nn_particles];
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, nn_particles, rn, -1.0, 1.0);

  #pragma omp simd
  for(int i = 0; i < nn_particles; i++) {
    pos[(a-1)*nn_particles + i] += dist_func(alpha, rn[i]);
  }

  return 0;
}