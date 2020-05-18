#ifndef MPCS_HPC_TICK_H
#define MPCS_HPC_TICK_H

#include <time.h>

// Convenience typedef
typedef struct timespec timespec_t;

// Returns a new timespec from CLOCK_MONOTONIC
timespec_t tick();

// Returns difference between two timespecs in nanosec
long tick_diff(timespec_t start, timespec_t end);

#endif // MPCS_HPC_TICK_H
