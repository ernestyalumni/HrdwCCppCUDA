//------------------------------------------------------------------------------
/// \file RuleOf5.cpp
/// \author Ernest Yeung
/// \brief Classes demonstrating copy and move constructors, assignments,
///   destructors.
///-----------------------------------------------------------------------------
#include "RuleOf5.h"

namespace Hierarchy
{

namespace RuleOf5
{

A::A(int n) :
  n_{n}
{}

// User-defined copy constructor.
A::A(const A& a) :
  n_{a.n_}
{}

} // namespace RuleOf5
} // namespace Hierarchy
