//------------------------------------------------------------------------------
/// \file AccessAndInheritance.h
/// \author Ernest Yeung
/// \brief Classes demonstrating access specifiers.
///-----------------------------------------------------------------------------
#ifndef _HIERARCHY_ACCESS_AND_INHERITANCE_H_
#define _HIERARCHY_ACCESS_AND_INHERITANCE_H_

namespace Hierarchy
{
namespace Access
{

// cf. https://en.cppreference.com/w/cpp/language/access

// "Ex" stands for Example
// Name of every class member (static, non-static, function, type, etc.) has an
// associated "member access." When name of member is used anywhere a program,
// its access is checked, and if it doesn't satisfy access rules, program
// doesn't compile (COMPILE-TIME check)
class ExPublicVsPrivate
{
  public: // declarations 

    // member "add" has public access
    // OK: private ExPublicVsPrivate::n_ can be access from
    // ExPublicVsPrivate::add
    void add(int x);

  private: // all declarations after this point are private
    // member "n_" has private access
    int n_ {0};
};


} // namespace Access
} // namespace Hierarchy

#endif // _HIERARCHY_ACCESS_AND_INHERITANCE_H_