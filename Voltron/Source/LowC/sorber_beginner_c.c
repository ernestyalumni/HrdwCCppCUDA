/* 
 * @url https://www.youtube.com/playlist?list=PL9IEJIKnBJjG5H0ylFAzpzs9gSmW_eICB
 * @ref Jacob Sorber. Beginner C Videos
 * @details
 * Example Compiling:
 * gcc hello.c
 * gcc hello.c -o hello
 * gcc sorber_beginner_c.c -o sorber_beginner_c
 *
 * Example Usage.
 * ./sorber_beginner_c
 * ./sorber_beginner_c Hello
 * ./sorber_beginner_c Hello My name is Jacob
 * ./sorber_beginner_c "Hello, my name is Jacob."
 */

// Preprocessor Directive
// Allows you to include code from another file.
#include <stdio.h>

/* Learning C: Basic Types (numbers,arrays,structs,pointers)
 * Jacob Sorber
 * @url https://youtu.be/mib3ahMbq_0
 */
#include <stdint.h>

/* Strings in C
 * Jacob Sorber
 * @url https://youtu.be/5TzFNouc0PE
 */
#include <string.h> // strlen

/* cf. Learn C in minutes (lesson 0)
 * @url https://youtu.be/SC8uWXmDJs4
 */
int add3(int a, int b, int c)
{
  int result = a + b + c;
  return result;
}

struct person {
  char name[50];
  int age;
  int height_in_inches;
};


/* cf. Jacob Sorber. Strings in C
 * @url https://youtu.be/5TzFNouc0PE
 */

int get_string_length(char* str)
{
  int offset = 0;
  while (str[offset] != 0) {
    offset++;
  }

  return offset;
}

void copy_string(char *from, char *to)
{
  int offset = 0;
  while (from[offset] != 0) {
    to[offset] = from[offset];
    offset++;
  }
  to[offset] = 0; // make sure it's NULL terminated.
}

/* Function Definition
* main is where your program will start, main is special.
 *
 * cf. Jacob Sorber. Getting Command-Line Arguments in C
 * https://youtu.be/6Dk8s0F2gow
 * argc, c - count
 * name of the executable, name of the "program", gets passed as one of the
 * arguments, and so for 
 * ./hello argc = 1
 * ./hello Hello argc = 2
 * ./hello Hello My name is Jacob argc = 6
 *
 * char **argv, double character pointer. Basically an array of strings.
 */
int main(int argc, char **argv) {

  /*
   * @url https://youtu.be/SC8uWXmDJs4
   * @brief Learn C in minutes (lesson 0)
   */

  // float, double struct, arrays, or pointers.
  // Can also initialize it to a value. If not, we don't know what it is and it
  // can give not good results.
  int mynumber = 0;
  int other;
  float arealnumber = 4.5;
  int result;

  mynumber = 24;
  other = 17;

  result = mynumber + other;

  printf("Hello World %i %i %i %f \n", mynumber, other, result, arealnumber);
  // 24 17 41 4.5000000

  printf("add3 = %i\n", add3(3, 67, 42)); // 112
  printf("add3 = %i\n", add3(mynumber, other, result)); // 82


  /* @brief Jacob Sorber. Learning C: Basic Types (numbers,arrays,structs,
   * pointers)
   * @url https://youtu.be/mib3ahMbq_0
   */
  {
    int x[50];

    x[0] = 5;
    x[3] = 500;

    printf("%i, %i\n",x[0],x[3]);

    struct person me;
    struct person you;

    me.age = 39;
    you.height_in_inches = 4;

    printf("%i, %i\n",me.age,you.height_in_inches);

    // Pointers store addresses, or they store locations in memory addresses.

    // x isn't an int, it's an address.
    int y = 7;
    int *p = &y; // ampersand gets the address.

    printf("%p, %i\n", p, *p); // 0x7ffdc3ee8b5c, 7

    // This says, change the int that this points to, to this value.
    *p = 14;
    printf("%p, %i\n", p, *p); // 0x7ffdc3ee8b5c, 14
  }

  /* @brief Jacob Sorber. Arrays, Pointers, and Why Arrays Start at Zero?
   * @url https://youtu.be/uT-YLEHwVS4
   */
  {
    // Allocated some space in array.
    int v[5] = {1,2,3,4,5};
    // Didn't allocate.
    int n = 9;

    // Immediately compiler complains because type mismatch.
    //int *p = n;

    int *p = v;

    printf("v[0] = %i\n",v[0]);
    printf("v[1] = %i\n",v[1]);
    printf("v[2] = %i\n",v[2]);
    printf("v[3] = %i\n",v[3]);
    printf("v[4] = %i\n",v[4]);

    // Pointers can act like arrays.

    // I can do this in arrays.
    v[3] = 7;

    p[4] = 9800;

    // Arrays act like pointers. Pointer arithmetic.

    printf("p = %p\n", p);

    // p + 2. (+8) It's adding +2(sizeof(int)). It's adding 2 times the size
    // of the thing that it's pointing to. +2(4) == 8
    //
    // Why?
    //
    // Pointers are arrays but in a different form, and C standard designed
    // assuming that you'll use pointers just like arrays.
    printf("p+2 = %p\n", p+2);
    
    printf("v[0] = %i\n",v[0]);
    printf("v[1] = %i\n",v[1]);
    printf("v[2] = %i\n",v[2]);
    printf("v[3] = %i\n",v[3]);
    printf("v[4] = %i\n",v[4]);

    printf("v[0] = %i\n",*p);
    // Moves the right number of bytes to get to the element in the array that
    // I want with pointer arithmetic (e.g. p+1)
    printf("v[1] = %i\n",*(p+1));
    printf("v[2] = %i\n",*(p+2));
    printf("v[3] = %i\n",v[3]);
    printf("v[4] = %i\n",v[4]);

    // Zero-based counting. Why?
    // element at the front, v[0]
    // v[0] is literally asking from the beginning of the array.

    *(v+2) = 49999; // same as v[2]
    printf("v[0] = %i\n",v[0]);
    printf("v[1] = %i\n",v[1]);
    printf("v[2] = %i\n",v[2]);

    // Arrays and pointers are different.

    // Compiler keeps track of size of v.
    //v[14] = 998899; // Clang will yell. Gcc won't.

    // Compiler can't help me with a pointer. But will seg fault. Because
    // it'll let you access memory at that address.
    // Pointer is just an address in memory.
    // EY: Didn't seg fault?
    //p[1400] = 998899;
  }

  /* @brief Jacob Sorber. Loops in C (while, do-while,for)
   * @url https://youtu.be/1vGfkBwIOXw
   */
  {
    int v[5] = {1,2,3,4,5};

    int counter = 0;
    while (counter < 5) {
      printf("v[%i] = %i\n", counter, v[counter]);
      counter = counter + 1;
    }

    // Do while loop.
    // Difference is that it checks the condition and the end of the loop.
    // First run always done. For while loop, it may not run at all.

    counter = 0;
    do {
      printf("v[%i] = %i\n", counter, v[counter]);
      counter = counter + 1;
    } while (counter < 4);

    for (int counter = 4; counter >= 0; counter --)
    {
      printf("v[%i] = %i\n", counter, v[counter]);
    }
  }

  /* @brief Jacob Sorber. Comments and Commenting in C
   * @url https://youtu.be/PP03QAsIij8
   */

  /* @brief Jacob Sorber. A Tour of C's Many Operators
   * @url https://youtu.be/PLnmboUvqq8
   */

  {
    int y;
    int x = 0;
    // Get value before x was changed.
    y = x++;
    printf("y= %i \n", y); // expect to be 0.
    // Get value after x was changed.
    y = ++x;
    printf("y= %i \n", y); // expect to be 2.

    // Bitwise operator.
    {
      int x = 5; // 0b101
      int y = 9;

      // 1001 (9), 0101 (5) & -> 0001
      int result = y & x;
      // 04x -> 0 - left pads number with zeroes (0) instead of spaces, where
      // padding is specified.
      // 4 (width) - min. number of characters to be printed.
      printf("result = %i %04x\n", result, result);

      // 1001 (9) | 0101 (5) -> 1101
      result = y | x;
      printf("result = %i %04x\n", result, result); // 13

      // XOR. Produces 1 if exactly 1 of the bits is 1.
      // 1001 (9) ^ 0101 (5) -> 1100
      result = y^x;
      printf("result = %i %04x\n", result, result); // 13

      // 1001 (9) ~ -> 0110 (6)
      result = ~y;
      printf("result = %i %04x\n", result, result); // 6

      // "Lose" 2 bits from the right.
      // Can think of it as divide by 2^2 = 4.
      result = y >> 2;
      printf("result = %i %04x\n", result, result); // 2

      result = y << 2;
      printf("result = %i %04x\n", result, result); // 36
    }
  }

  /* @brief Jacob Sorber. Strings in C
   * @url https://youtu.be/5TzFNouc0PE
   * @details Store a length, or some way to tell it's an end.
   * In C, it's 0, null character. Without, it goes on forever.
   */
  {
    // TODO: do gdb to show that this text inside double quotes gets stored
    // in a string, as .asciz
    printf("Hello World!\n");

    // Pointers and arrays are the same.
    // Size of these strings are 13 bytes (for "Hello World!") because must
    // count null-terminating character.
    char *str1 = "Hello World! I'm string 1";
    char str2[] = "Hello World!";

    printf("%s",str1);
    printf("%s",str2);

    /* In lookup table (e.g. www.lookupTables.com), characters under decimal 32
     * are control characters, equal and above decimal 32 are printable
     * characters.
     */

    // Use Octal 012 not Decimal 10 because you need octal here, to specify new
    // line. 12 is the character code, but in octal. 10 dec = 12 oct.
    printf("s\12",str1);
    printf("s\12",str2);

    printf("%s\xA",str1);
    printf("%s\xA",str2);

    printf("%s\n",str1);
    printf("%s\n",str2);
    printf("Print a backslash \\ \n");

    // String manipulation. We're working with arrays with numbers.
    
    printf("%s has length %d bytes\n", str1, get_string_length(str1));
    printf("%s has length %d bytes\n", str1, strlen(str1));

    char newstring[500];
    copy_string(str1, newstring);
    printf("%s\n",newstring);

    char newstring2[500];
    // sorry strcpy's args go the other way.
    strcpy(newstring2, str1);
    printf("%s\n",newstring2);


    /* Lesson one, if you type 
     * 
     * man string
     *
     * in the command prompt you get list of all string functions to use.
     *
     * Homework. String reverse. e.g. "future video"
     */
  }


  /* @brief Jacob Sorber. Getting Command-Line Arguments in C
   * @url https://youtu.be/6Dk8s0F2gow
   * @details 
   *
   * For command line arguments, if you want to include spaces in a single
   * string, put them in double quotes (""), e.g.:
   *
   * ./hello "Hello, my name is Jacob."
   * ./sorber_beginner_c "Hello, my name is Jacob"
   */
  {
    printf("Hello World - %d\n", argc);

    for (int i=0; i < argc; i++) {
      printf("arg %d - %s\n",i,argv[i]);
    }    
  }

  return 0;
}