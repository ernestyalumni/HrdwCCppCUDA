/*******************************************************************************
 * @ref MIT 6.172 Fall 2018 Hwk 1, c-primer
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define PRINT_SIZE_1(T_STR, T) \
  printf("size of %s : %zu bytes \n", T_STR, sizeof(T));

// char * argv is an array of pointers to char.
int main(int argc, char * argv[]) {
  if (argc > 1)
  {
    for (int i = 1; i < argc; ++i)
    {
      printf("value: %s, %zu Size bytes \n", argv[i], sizeof(argv[i]));
    }
  }

  int i = 5;
  // The & operator here gets the address of i and stores it into pi
  int * pi = &i;

  // The * operator here dereferences pi and stores the value -- 5 --
  // into j.
  int j = *pi;
  PRINT_SIZE_1("deferenced int", &j);

  char c[] = "6.172";
  char * pc = c; // Valid assignment: c acts like a pointer to c[0] here.
  char d = *pc;

  printf("char d = %c\n", d); // This prints "6"

  // Compound types are read right to left in C.
  // pcp is a pointer to a pointer to a char, meaning that
  // pcp stores the address of a char pointer.

  // Originally,
  // char ** pcp;
  char* *pcp; // pointer to a pointer to a char.

  PRINT_SIZE_1("\n char** pcp \n", pcp);

  pcp = argv; // Why is this assignment valid?
  // argv is an array of char ptrs, but itself points to first char*.

  PRINT_SIZE_1("\n After assignment: char** pcp \n", pcp);

  const char* pcc = c; // pcc is a pointer to char constant.
 
  PRINT_SIZE_1("\n const char* pcc \n", pcc);

  char const* pcc2 = c; // What is the type of pcc2?
  // pcc2 is a const pointer to a char. The pointer itself can't change.

  PRINT_SIZE_1("\n char const* pcc2 \n", pcc2);

  //----------------------------------------------------------------------------

  // For each of the following, why is the assignment:

  // error: assignment of read only location.
  //*pcc = '7'; // invalid?

  pcc = *pcp; // The char pointer on the right gets assigned to a pointer to a
  // char constant. valid.

  pcc = argv[0]; // valid?
  // argv is an array of char pointers, so a right-hand side char pointer is
  // assigned to a pointer to a char constant.


  //----------------------------------------------------------------------------

  char * const cp = c; // cp is a const pointer to char.
  // For each of the following, why is assignment:

  //cp = *pcp; // invalid?
  // Char pointer on the right is being assigned to something that is const.
  // error: assignment of read-only variable.
  //cp = *argv; // invalid?
  // Char pointer on right begin assigned to something that's const.
  *cp = '!'; // valid?
  // cp points to a char. cp has address of the char. We can go to that address
  // and directly change the value that is stored at that address.

  printf("char c[] = %c %c %c %c \n", c[0], c[1], c[2], c[3]); // This prints
  // '!', '.', '1', '7'

  //----------------------------------------------------------------------------

  const char * const cpc = c; // cpc is a const pointer to char const
  // For each of the following, why is the assignment:
  //cpc = *pcp; // invalid? char pointer gets assigned to const pointer.
  //cpc = argv[0]; // invalid? Char pointer gets assigned to const pointer.
  //*cpc = '@'; // invalid? Const char can't be changed.
}