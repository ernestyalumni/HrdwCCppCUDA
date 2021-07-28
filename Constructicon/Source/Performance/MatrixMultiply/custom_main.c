#include <stdio.h>
#include <time.h>

#include "fasttime.h"
#include "matrix_multiply.h"

#define PRINT_SIZE_1(T_STR, T) \
  printf("size of %s : %zu bytes \n", T_STR, sizeof(T));

int main() {
 
  printf("\n ----- MIT 6.172 Fall 2018 Matrix Multiply Custom Main ----- \n");

  fasttime_t test_get_time_result = gettime();

  printf(
    "\n test_get_time_result components : %zu , %zu \n",
    test_get_time_result.tv_sec,
    test_get_time_result.tv_nsec);

  PRINT_SIZE_1("fasttime_t", fasttime_t);
  PRINT_SIZE_1("timespec time_t", test_get_time_result.tv_sec);
  PRINT_SIZE_1("timespec long", test_get_time_result.tv_nsec);

  fasttime_t test_get_time_result_1 = gettime();

  printf(
    "\n test tdiff: %f \n",
    tdiff(test_get_time_result, test_get_time_result_1));

}