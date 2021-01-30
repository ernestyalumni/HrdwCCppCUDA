/* 
 * @url https://www.youtube.com/playlist?list=PL9IEJIKnBJjG5H0ylFAzpzs9gSmW_eICB
 * @ref Jacob Sorber. Beginner C Videos
 * @details
 * Example Usage:
 * gcc hello.c
 * gcc hello.c -o hello
 * gcc sorber_beginner_c.c -o sorber_beginner_c
 */

// Preprocessor Directive
// Allows you to include code from another file.
#include <stdio.h>

/* Learning C: Basic Types (numbers,arrays,structs,pointers)
 * Jacob Sorber
 * @url https://youtu.be/mib3ahMbq_0
 */
#include <stdint.h>

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


// Function Definition
// main is where your program will start, main is special.
int main() {

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


  return 0;
}