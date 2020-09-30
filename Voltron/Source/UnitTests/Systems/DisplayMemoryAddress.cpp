//------------------------------------------------------------------------------
/// \file DisplayMemoryAddress
/// \ref https://www.tenouk.com/cpluscodesnippet/cplusarraypointermemoryaddress.html
/// \brief A program that uses pointers to print the array elements
//------------------------------------------------------------------------------
#include <iostream>

int main()
{
  // declare and initialize an array nums.
  int nums[] = {92, 81, 70, 69, 58};

  std::cout << "\nArray's element Memory address";
  std::cout << "\n------------------------------";

  // Using for loop, displays the elements of nums and their respective memory
  // address.

  for (int dex {0}; dex < 5; ++dex)
  {
    std::cout << "\n\t" << *(nums + dex) << "\t\t" << (nums + dex);
    std::cout << "\n------------------------------";
  }

  return 0;
}

// Example Output:
// Array's element Memory address
// ------------------------------
//   92    0x7ffe23411140
// ------------------------------
//   81    0x7ffe23411144
// ------------------------------
//   70    0x7ffe23411148
// ------------------------------
//   69    0x7ffe2341114c
// ------------------------------
//   58    0x7ffe23411150
// ------------------------------