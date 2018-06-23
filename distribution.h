#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cmath>
#include <omp.h>
#include <mkl.h>
const float delta_max = 1.0f;

#pragma omp declare simd
float dist_func(const float alpha, float rn);

int iterate_particles(const float alpha, unsigned int i, short int a, int total_length, VSLStreamStatePtr rnStream, int nn_particles, float *pos);
#endif