//------------------------------------------------------------------------------
/// \file Stack.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating stack.
/// \details
///
/// \ref https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
/// \ref https://www.techiedelight.com/stack-implementation-in-cpp/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_STACK_H
#define DATA_STRUCTURES_STACK_H

namespace DataStructures
{
namespace Stack
{

template <typename T>
class Stack
{
  public:


    T peek()
    {
      if (top_ < 0)
      {

      }
    }

  private:

    long top_;
};

} // namespace Stack
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACK_H