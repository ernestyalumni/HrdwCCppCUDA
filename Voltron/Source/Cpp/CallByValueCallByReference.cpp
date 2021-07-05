#include "CallByValueCallByReference.h"

#include <iostream>

using std::cout;
using std::endl;

namespace Cpp
{

namespace CallByValue
{

void my_function(int x)
{
  x = 50;

  cout << "Value of x from my_function: " << x << endl;
}

int call_by_value(int x)
{
  x = 50;

  cout << "Value of x from call_by_value: " << x << endl;

  return x;
}

} // namespace CallByValue

namespace CallByReference
{

void my_function(int& x)
{
  x = 50;

  cout << "Value of x from my_function: " << x << endl;
}

int call_by_ref(int& x)
{
  x = 50;
  cout << "Value of x from call_by_ref: " << x << endl;

  return x;
}

const int& call_by_const_ref(const int& x)
{
  // Fails at compile-time. Assignment of read-only reference.
  //x = 50;

  cout << "Value of x from call_by_const_ref: " << x << endl;

  return x;  
}

int const& call_by_ref_const(int const& x)
{
  // Fails at compile-time, error: assignment of read-only reference
  //x = 50;

  cout << "Value of x from call_by_ref_const: " << x << endl;

  return x;  
}

} // namespace CallByReference

} // namespace Cpp
