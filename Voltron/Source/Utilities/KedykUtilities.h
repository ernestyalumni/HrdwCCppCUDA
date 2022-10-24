#ifndef UTILITIES_KEDYK_UTILITIES_H
#define UTILITIES_KEDYK_UTILITIES_H

#include <cstddef>

namespace Utilities
{
namespace Kedyk
{

//------------------------------------------------------------------------------
/// \ref 5.2 Utility Functions, Implementing Useful Algorithms in C++. Kedyk.
/// \url https://github.com/dkedyk/ImplementingUsefulAlgorithms/blob/master/Utils/Debug.h
//------------------------------------------------------------------------------
template <typename INT_T, typename UNSIGNED_INT_T>
INT_T ceiling(const UNSIGNED_INT_T n, const INT_T divisor)
{
  return n / divisor + static_cast<bool>(n % divisor);
}

//------------------------------------------------------------------------------
/// Convenience wrappers around placement new and delete. Remember that these
/// don't call ctors and dtors and must be used in pairs.
//------------------------------------------------------------------------------
template <typename ITEM_T>
ITEM_T* raw_memory(const std::size_t n)
{
  return (ITEM_T*)::operator new(sizeof(ITEM_T) * n);
}

void raw_delete(void* array)
{
  ::operator delete(array);
}

template <typename ITEM_T>
void raw_destruct(ITEM_T* array, const std::size_t size)
{
  for (std::size_t i {0}; i < size; ++i)
  {
    array[i].~ITEM_T();
    raw_delete(array);
  }
}

} // namespace Kedyk
} // namespace Utilities

#endif // UTILITIES_KEDYK_UTILITIES_H