#include "tick.h"

timespec_t tick() {
  timespec_t res;
  clock_gettime(CLOCK_MONOTONIC, &res);
  return res;
}

long tick_diff(timespec_t start, timespec_t end) {
  return 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec -
         start.tv_nsec;
}
