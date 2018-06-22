#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H
#include <cmath>
#include <omp.h>
const float delta_max = 1.0f;

#pragma omp declare simd
float dist_func(float buffer, const float alpha, float rn);

#pragma omp declare simd
bool partition_func(int *status, int n_steps, int number_split);

#endif