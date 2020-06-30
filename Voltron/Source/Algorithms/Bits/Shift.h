//------------------------------------------------------------------------------
/// \file Shift.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Utilities to deal with shift operators.
/// \ref https://stackoverflow.com/questions/13221369/logical-shift-right-on-signed-data/57608876#57608876
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_BITS_SHIFT_H
#define ALGORITHMS_BITS_SHIFT_H

#include <type_traits>

namespace Algorithms
{
namespace Bits
{

// cf. https://stackoverflow.com/questions/13221369/logical-shift-right-on-signed-data/57608876#57608876
// cf. https://stackoverflow.com/questions/13221369/logical-shift-right-on-signed-data
template <class T>
inline T logical_right_shift(T t1, T t2)
{
  // https://en.cppreference.com/w/cpp/types/make_unsigned
  // make_unsigned - If T is integral or enum type, provides member typedef
  // type which is unsigned integer type corresponding to T, with same
  // cv-qualifiers.
  //
  // https://www.geeksforgeeks.org/static_cast-in-c-type-casting-operators/
  // https://stackoverflow.com/questions/332030/when-should-static-cast-dynamic-cast-const-cast-and-reinterpret-cast-be-used
  // static_cast returns a value of type new_type. Forces 1 data type to be
  // converted into another data type. Compile time cast.
  // Does implicit conversions, can also call explicit conversion functions,
  // casts through inheritance hierarchies, unncessary when casting upwards,
  // undefined behavior down hierarchy through virtual inheritance.
  return static_cast<typename std::make_unsigned_t<T>>(t1) >> t2;
}

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_SHIFT_H
