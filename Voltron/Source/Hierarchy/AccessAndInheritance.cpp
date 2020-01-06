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

int ExPublicVirtualMethod::f()
{
  return 42;
}

int ExPrivateVirtualMethod::f()
{
  return 43;
}

void ProtectedMemberAccessBase::g(
  ProtectedMemberAccessBase& b,
  ProtectedMemberAccessDerived& d) // member function of
  // ProtectedMemberAccessBase
{
  ++i_; // ok
  ++b.i_;
  ++d.i_;
}

void non_member_non_friend_access(
  ProtectedMemberAccessBase& b,
  ProtectedMemberAccessDerived& d) // non-member non-friend
{
  // COMPILE-TIME error, error: no access from non-member, protected
  // ++b.i_ // error: no access from non-member
  // COMPILE-TIME error, error: no access from non-member, protected
  // ++d.i_ // error: no access from non-member
}

} // namespace Access
} // namespace Hierarchy
