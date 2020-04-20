//------------------------------------------------------------------------------
/// \file FloatingPoint.cpp
/// \author Ernest Yeung
/// \brief Dealing with floating point representations.
/// \ref 
///-----------------------------------------------------------------------------
#include "FloatingPoint.h"

#include "Cpp/Utilities/SuperBitSet.h"

#include <climits> // CHAR_BIT

using Cpp::Utilities::SuperBitSet;

namespace Utilities
{

namespace Conversions
{

namespace WithUnion
{

// Need to put specialization in .cpp source file or you end up violating one
// definition rule. Intuitively, once fully specialized, it doesn't depend on
// template parameter anymore. And it was not made inline.
// cf. https://stackoverflow.com/questions/4445654/multiple-definition-of-template-specialization-when-using-different-objects

template <>
SuperBitSet<sizeof(float) * CHAR_BIT>
  floating_point_to_bitset<float, sizeof(float)>(const float x)
{
  const std::size_t size {sizeof(float) * CHAR_BIT};

  FloatingPointToBitsInUnion::FloatUIntDirectSum data;
  data.input_ = x;
  return SuperBitSet<size>{data.output_};
}

template <>
SuperBitSet<sizeof(double) * CHAR_BIT>
  floating_point_to_bitset<double, sizeof(double)>(const double x)
{
  const std::size_t size {sizeof(double) * CHAR_BIT};

  FloatingPointToBitsInUnion::DoubleULongDirectSum data;
  data.input_ = x;
  return SuperBitSet<size>{data.output_};
}

} // namespace WithUnion

namespace WithMemCpy{

} // namespace WithMemCpy

} // namespace Conversions
} // namespace Utilities