//-----------------------------------------------------------------------------
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/2/Dynamic_stack/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_STACKS_DYNAMIC_STACK_H
#define DATA_STRUCTURES_STACKS_DYNAMIC_STACK_H

#include "Stack.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace DataStructures
{
namespace Stacks
{

namespace AsHierarchy
{

template <typename T>
class DynamicStack : DataStructures::Stacks::AsHierarchy::Stack<T>
{
  public:

    DynamicStack(const std::size_t N = 10);

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    DynamicStack(const DynamicStack&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    DynamicStack& operator=(const DynamicStack&);

    //--------------------------------------------------------------------------
    /// \brief Move assignment.
    //--------------------------------------------------------------------------
    DynamicStack& operator=(DynamicStack&&);

    virtual ~DynamicStack();

    virtual T top() const override;
    virtual std::size_t size() const override;
    virtual bool is_empty() const override;
    std::size_t capacity() const;

    void swap(DynamicStack& other);

    virtual void push(const T item) override;

    virtual T pop() override;

    //--------------------------------------------------------------------------
    /// \details Worst case run time O(N) for N copies.
    //--------------------------------------------------------------------------
    void clear();

  private:

    void double_capacity()
    {
      // We must have access to this new T[N] where N is the new capacity. So
      // we must store the address returned by new in a local variable, say
      // tmp_array.
      T* tmp_array = new T[2 * array_capacity_];

      // Next, the values must be copied over.
      // Requires N copies, run time is O(N).
      std::copy(array_, array_ + array_capacity_, tmp_array);

      // The memory for the original array must be deallocated.
      delete [] array_;

      // Finally, the appropriate member variables must be reassigned.
      array_ = tmp_array;

      array_capacity_ *= 2;
    }

    std::size_t entry_count_;
    std::size_t initial_capacity_;
    std::size_t array_capacity_;

    T* array_;
};

template <typename T>
DynamicStack<T>::DynamicStack(const std::size_t N):
  entry_count_{0},
  initial_capacity_{N},
  array_capacity_{N}
{
  array_ = new T[array_capacity_];
}

template <typename T>
DynamicStack<T>::DynamicStack(const DynamicStack& stack):
  entry_count_{stack.entry_count_},
  initial_capacity_{stack.initial_capacity_},
  array_capacity_{stack.array_capacity_},
  array_{new T[array_capacity_]}
{
  // The above initialization copies the values of the appropriate member
  // variables and allocate memory for the data structure; however, you must
  // still copy the stored objects.

  std::copy(
    std::begin(stack.array_),
    std::end(stack.array_),
    std::begin(array_));
}

template <typename T>
DynamicStack<T>& DynamicStack<T>::operator=(const DynamicStack& stack)
{
  entry_count_ = stack.entry_count_;
  initial_capacity_ = stack.initial_capacity_;
  array_capacity_ = stack.array_capacity_;

  std::copy(
    std::begin(stack.array_),
    std::end(stack.array_),
    std::begin(array_));
}

template <typename T>
DynamicStack<T>& DynamicStack<T>::operator=(DynamicStack&& stack)
{
  // Swap.
  std::swap(entry_count_, stack.entry_count_);
  std::swap(initial_capacity_, stack.initial_capacity_);
  std::swap(array_capacity_, stack.array_capacity_);
  std:swap(array_, stack.array_); 
}

template <typename T>
DynamicStack<T>::~DynamicStack()
{
  delete [] array_;
}

template <typename T>
T DynamicStack<T>::top() const
{
  if (is_empty())
  {
    throw std::runtime_error("Called top on empty Dynamic Stack");
  }

  // If there are N objects in the stack, the last is located at index N - 1.
  return array_[entry_count_ - 1];
}

template <typename T>
std::size_t DynamicStack<T>::size() const
{
  return entry_count_;
}

template <typename T>
bool DynamicStack<T>::is_empty() const
{
  return entry_count_ == 0;
}

template <typename T>
void DynamicStack<T>::push(const T item)
{
  if (entry_count_ == array_capacity_)
  {
    double_capacity();
  }

  array_[entry_count_] = item;
  ++entry_count_;
}

//------------------------------------------------------------------------------
/// \ref 3.02.Stacks.pptx, Slide 22, D.W. Harder, U. of Waterloo.
/// \details Removing an object simply involves reducing the size.
/// It's invalid to assign the last entry to "0".
/// By decreasing the size, the previous top of the stack is now at the location
/// stack size or entry count.
//------------------------------------------------------------------------------
template <typename T>
T DynamicStack<T>::pop()
{
  if (is_empty())
  {
    throw std::runtime_error("Called pop on empty DynamicStack");
  }

  --entry_count_;

  return array_[entry_count_];
}


template <typename T>
void DynamicStack<T>::clear()
{
  entry_count_ = 0;
}

} // namespace AsHierarchy

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_DYNAMIC_STACK_H