#include "reverse_string.h"

#include <stdio.h> // printf

int main()
{
  // N = 0
  {
    // char *str = "" or char *str = "stackoverflow" creates a read-only string
    // literal. Modifying it is undefined behavior. Use an array instead.
    // cf. https://stackoverflow.com/questions/15374529/segmentation-fault-when-swapping-pointer-values-while-reversing-a-string

    char str[] = "";

    printf("str before reverse: %s \n", str);
    reverse_string(str);
    printf("str after reverse:  %s \n", str);
  }

  // N = 1
  {
    char str[] = "a";
    printf("str before reverse: %s \n", str);
    reverse_string(str);
    printf("str after reverse:  %s \n", str);    
  }
  
  // N = 2
  {
    // If you do char *str = "aB", this creates a read-only string literal, str
    // points to read-only memory. Cannot be modified. 
    // cf. https://stackoverflow.com/questions/12795850/string-literals-pointer-vs-char-array

    char str[] = "aB";
    printf("str before reverse: %s \n", str);

    // Example code breaking down steps.    
    //char temp = str[0];

    // This will seg fault if char* str = "aB". This is because char *str = "aB"
    // is a read-only string literal. Then there's memory access violation if
    // you try to access memory at str[0] to modify it.
    //str[0] = str[1];

    reverse_string(str);
    printf("str after reverse:  %s \n", str);    
  }
  
  // N
  {
    char str[] = "future video";
    printf("str before reverse: %s \n", str);
    reverse_string(str);
    printf("str after reverse:  %s \n", str);        
  }
}