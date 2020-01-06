//------------------------------------------------------------------------------
/// \file AccessAndInheritance.cpp
/// \author Ernest Yeung
/// \brief Classes demonstrating access specifiers.
///-----------------------------------------------------------------------------
#include "AccessAndInheritance.h"

namespace Hierarchy
{

namespace Access
{

void ExPublicVsPrivate::add(int x)
{
  // member "add" has public access
  n_ += x; // OK: private ExPublicVsPrivate::n_ can be access from
  // ExPublicVsPrivate::add  
}


} // namespace Access
} // namespace Hierarchy
