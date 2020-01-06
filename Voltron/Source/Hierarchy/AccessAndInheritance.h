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

// cf. https://en.cppreference.com/w/cpp/language/access "In detail"

// To grant access to ADDITIONAL functions or classes, to protected or private
// members, a friendship declaration may be used.

// Accessibility applies to all names with no regard to origin, so name
// introduced by typedef or using declarations is checked, not name it refers
// to.
class ExPublicUsing
{
  class B {}; // B is private in A

  public:
    using BB = B; // BB is public
};

// Member access check is last step after any given language construct is
// interpreted. Intent is replacing any private with public never alters
// behavior of program.

// Access checking for names used in default function arguments as well as
// default template parameters is performed at point of declaration, not at the
// point of use.

// Access rules for names of virtual functions checked at call point using type
// of expression used to denote object for which member function is called.
struct ExPublicVirtualMethod
{
  // f is public in ExVirtualBaseStruct
  virtual int f();
};


class ExPrivateVirtualMethod : public ExPublicVirtualMethod
{
  private:

    int f(); // f is private in ExPrivateVirtualMethod
};

// When a member is redeclared within same class, it must do so under the same
// member access:
class MemberRedeclared
{
  public:
    class A; // MemberRedeclared::A is public

  private:
    // error at COMPILE-TIME
    //class A {}; // error: cannot change access.
  public:

    class A {};
};

// A name that's private according to unqualified name lookup, may be accessible
// through qualified name lookup:
class NameLookupA
{};

class NameLookupB : private NameLookupA
{};

class NameLookupC : public NameLookupB
{
  //NameLookupA* p; // error: unqualified name lookup finds NameLookupA as the
  // private base of NameLookupB
  Access::NameLookupA* q; // OK, qualified name lookup finds namespace-level declaration
  // (!!!)
};

// Public members form a part of public interface of class (other parts of
// public interface are non-member functions found by ADL).
// A public member of a class is accessible everywhere.
class PublicMemberAccessS
{
  public: // n, f, E, A, B, C, U are public members
    int n_;
    static void f()
    {}
    enum E {A, B, C};
    struct U {};
};

// A protected member of a class Base can only be accessed 
// 1) by members and friends of Base
// 2) by members and friends (until C++17) of any class derived from Base, but
// only when operating on an object of a type that's derived from Base
// (including this)

struct ProtectedMemberAccessBase
{
  protected:
    int i_;
  private:
    void g(
      ProtectedMemberAccessBase& b,
      struct ProtectedMemberAccessDerived& d);
};

struct ProtectedMemberAccessDerived : ProtectedMemberAccessBase
{
  // member function of a derived class
  void f(ProtectedMemberAccessBase& b, ProtectedMemberAccessDerived& d)
  {
    ++d.i_; // okay; the type of d is ProtectedMemberAccessDerived
    ++i_; // okay, the type of the implied '*this' is
    //ProtectedMemberAccessDerived
    // ++b.i_; // error: can't access a protected member through
    // ProtectedMemberAccessBase (Otherwise it would be possible to change
    // other derived classes, like a hypothetical Derived2, base
    // implementation)
  }
};

void non_member_non_friend_access(
  ProtectedMemberAccessBase& b,
  ProtectedMemberAccessDerived& d);

// When a pointer to a protected member is formed, it must use a derived class
// in its declaration
struct PointerMemberDerived : ProtectedMemberAccessBase
{
  void f()
  {
    // COMPILE-TIME error
    // error: must name using derived class PointerMemberDerived
    //int ProtectedMemberAccessBase::* ptr1 = &ProtectedMemberAccessBase::i_;

    int ProtectedMemberAccessBase::* ptr = &PointerMemberDerived::i_;
  }
};

// A private member of a class can only be accessed by members and friends of
// that class, regardless of whether the members are on the same or different
// instances:
class PrivateAccessS
{
  private:

    int n_; // PrivateAccessS::n_ is private

  public:
    PrivateAccessS() :
      n_{10}
    {} // this->n_ is accessible in PrivateAccessS::PrivateAccessS

    PrivateAccessS(const PrivateAccessS& other) :
      n_{other.n_} // other.n_ is accessible in PrivateAccessS::PrivateAccessS
    {}

};

} // namespace Access
} // namespace Hierarchy

#endif // _HIERARCHY_ACCESS_AND_INHERITANCE_H_