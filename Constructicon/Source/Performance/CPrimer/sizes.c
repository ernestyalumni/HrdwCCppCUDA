/*******************************************************************************
 * @ref MIT 6.172 Fall 2018 Hwk 1, c-primer
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Solution 1: Define a macro to avoid copy pasting this code as such:
// e.g. PRINT_SIZE("int", int);
//      PRINT_SIZE("short", short);

#define PRINT_SIZE_1(T_STR, T) \
  printf("size of %s : %zu bytes \n", T_STR, sizeof(T));

// Alternatively, you can use stringification
// () so that
// you can write
// e.g. PRINT_SIZE(int);
//      PRINT_SIZE(short);

/*******************************************************************************
 * When macro parameter is used with a leading #, preprocessor replaces it with
 * literal text of actual argument, converted to string constant. Unlike normal
 * parameter replacement, argument isn't macro-expanded first. This is called
 * stringification.
 ******************************************************************************/ 

#define PRINT_SIZE_STRINGIFY(T) \
  printf("size of " #T " : %zu bytes \n", sizeof(T));

int main() {

  // Please print the sizes of the following types:
  // int, short, long, char, float, double, unsigned int, long long
  // uint8_t, uint16_t, uint32_t, and uint64_t, uint_fast8_t,
  // uint_fast16_t, uintmax_t, intmax_t, __int128, and student

  printf("size of %s : %zu bytes \n", "int", sizeof(int));

  PRINT_SIZE_1("short", short); // 2 bytes

  PRINT_SIZE_STRINGIFY(long); // 8 bytes

  PRINT_SIZE_STRINGIFY(char); // 1 bytes

  PRINT_SIZE_STRINGIFY(float); // 4 bytes

  PRINT_SIZE_STRINGIFY(double); // 8 bytes

  PRINT_SIZE_STRINGIFY(unsigned int); // 4 bytes

  PRINT_SIZE_STRINGIFY(long long); // 8 bytes

  PRINT_SIZE_STRINGIFY(uint8_t); // 1 bytes

  PRINT_SIZE_STRINGIFY(uint16_t); // 2 bytes

  PRINT_SIZE_STRINGIFY(uint32_t); // 4 bytes

  PRINT_SIZE_STRINGIFY(uint64_t); // 8 bytes

  PRINT_SIZE_STRINGIFY(uint_fast8_t); // 1 bytes

  PRINT_SIZE_STRINGIFY(uint_fast16_t); // 2 bytes

  PRINT_SIZE_STRINGIFY(uintmax_t); // 8 bytes

  PRINT_SIZE_STRINGIFY(intmax_t); // 8 bytes

  // __int128 is a clang C extension and not part of standard C.
  PRINT_SIZE_STRINGIFY(__int128); // 16 bytes


  // Composite types have sizes too
  typedef struct {
    int id;
    int year;
  } student;

  student you;
  you.id = 12345;
  you.year = 4;


  // Array declaration. Use your macro to print the size of this.
  int x[5];

  PRINT_SIZE_1("x, int[5] ", x); // 20 bytes

  PRINT_SIZE_1("pointer to x, int[5] ", &x); // 8 bytes

  // You can just use your macro here instead: PRINT_SIZE("student", you);
  printf("size of %s : %zu bytes \n", "student", sizeof(you));

  PRINT_SIZE_1("pointer to student", &you); // 8 bytes

  return 0;
}