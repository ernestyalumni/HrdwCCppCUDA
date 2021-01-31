/* Jacob Sorber. Getting Command-Line Arguments in C
 * https://youtu.be/6Dk8s0F2gow
 *
 * Example Usage
 * ./hello Hello 1 2 53 45.0
 * ./sorber_get_command_line Hello 1 2 53 45.0
 */

#include <stdio.h>
#include <stdlib.h> // atoi, atof

int main(int argc, char **argv) {

  printf("Hello World - %d\n", argc);

  for (int i=0; i < argc; i++) {

    // atoi - short for ascii to int.
    // cf. https://en.cppreference.com/w/cpp/string/byte/atoi
    // If converted value falls out of range of corresponding return type,
    // return value undefined. If no conversion can be performed, 0 returned.
    printf("arg %d - %s, %i, %f\n",i,argv[i], atoi(argv[i]), atof(argv[i]));
  }
}