// \file CoderByteChallenges.cpp

#include <cassert>
#include <iostream>
#include <string>

// https://coderbyte.com/editor/First%20Factorial:Cpp
int FirstFactorial(int num)
{
  assert(num > 0);

  if (num == 1)
  {
    return 1;
  }

  return num * FirstFactorial(num - 1);  
}

int main(void)
{
  // https://coderbyte.com/editor/First%20Factorial:Cpp
  {
    std::cout << "\n First Factorial begins \n";

    std::cout << " FirstFactorial(1) : " << FirstFactorial(1) << "\n";
    std::cout << " FirstFactorial(2) : " << FirstFactorial(2) << "\n";
    std::cout << " FirstFactorial(3) : " << FirstFactorial(3) << "\n";
    std::cout << " FirstFactorial(4) : " << FirstFactorial(4) << "\n"; // 24
    std::cout << " FirstFactorial(8) : " << FirstFactorial(8) << "\n"; // 40320

    std::cout << "\n First Factorial ends \n";

  }


  return 0;
}