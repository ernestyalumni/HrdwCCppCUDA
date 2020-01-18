// \file Entrevue.cpp

#include <cassert>
#include <iostream>
#include <string>

int f1()
{
  std::cout << "\n f1 \n";
  return 31;
}

int main(void)
{
  {
    std::cout << "\n \n Entrevue begin" << "\n";    
  }

  // https://coderbyte.com/editor/First%20Factorial:Cpp
  {
    f1();

  }

  {
    std::cout << "\n \n Entrevue end" << "\n";    
  }


  return 0;
}