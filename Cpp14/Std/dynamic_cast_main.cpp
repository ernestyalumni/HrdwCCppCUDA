//------------------------------------------------------------------------------
/// \file dynamic_cast_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Safely converts pointers and references to classes up, down, and
///   sideways along the inheritance hierarchy.
/// \ref https://en.cppreference.com/w/cpp/language/dynamic_cast
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
///
/// \details Syntax:
/// dynamic_cast<new_type>(expression)
///
/// \tparam new_type - pointer to complete class type, reference to complete
/// class type, or pointer to (optionally cv-qualified) void
/// \param expression - glvalue of complete class type if new_type is a 
/// reference, prvalue of a pointer to complete class type if new_type is a ptr.
///
/// If cast is successful, dynamic_cast returns value of type new_type. If cast
/// fails and new_type is a ptr type, it returns a null ptr of that type.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 dynamic_cast_main.cpp -o dynamic_cast_main_main
//------------------------------------------------------------------------------

#include <iostream>

//------------------------------------------------------------------------------
/// \url https://en.cppreference.com/w/cpp/language/dynamic_cast
/// \details Only the following conversions can be done with dynamic_cast,
/// except when such conversions would cast away constness or volatility:
///
/// 1. If type of expression is exactly new_type or less cv-qualified version of
/// new_type, result is value of expression, with type new_type
/// i.e. dynamic_cast can be used to add constness; an implicit conversion and
/// static_cast can perform this conversion as well.
/// i.e.
///   new_type == type(expression)
///
/// 2. If value of expression is null ptr value, result is null ptr value of
/// type new_type.
/// i.e. if expression = nullptr,
/// dynamic_cast<new_type>(expression) |-> new_expression; where 
///   type(new_expression) = new_type and new_expression = nullptr // in value
///
/// 3. If new_type is ptr or ref to Base, and type of expression is ptr or ref
/// to Derived, where Base is a unique, accessible base class of Derived, result
/// is a ptr or ref to Base class subobject within the Derived object pointed or
/// identified by expression.
///   (Note: an implicit conversion and static_cast can perform this conversion
///   as well.)
/// i.e.
/// dynamic_cast<Base>(expression) |-> new_expression; where 
/// type(expression) = Derived
///
/// 4. If expression is ptr to polymorphic type, and new_type is ptr to void,
/// result is ptr to most derived object pointed or referenced by expression.
/// 5. If expression is ptr or ref to polymorphic type Base, and new_type is ptr
/// or ref to type Derived, a run-time check is performed: // EY RUN-TIME check!
///   a) Most derived object pointed/identified by expression is examined. If,
/// in that object, expression points/refers to a public base of Derived, and if
/// only 1 subobject or Derived type is derived from subobject pointed/
/// identified by expression, then result of cast points/refers to that Derived
/// subobject. (this is known as a "downcast.")
///   b) Otherwise, if expression points/refers to public base of most derived
/// object, and simultaneously, most derived object has unambiguous public base
/// class of type Derived, result of cast points/refers to that Derived (This is
/// known as "sidecast")
//------------------------------------------------------------------------------

struct V
{
  virtual void f() // must be polymorphic to use runtime-checked dynamic_cast
  {};
};

struct A : virtual V
{};

struct B : virtual V
{
  B(V* v, A* a)
  {
    // casts during construction (see the call in the constructor of D below)
    dynamic_cast<B*>(v); // well-defined: v of type V*, V base of B, results in B*
    //dynamic_cast<B*>(a); // undefined behavior: a has type A*, A not a base of B
  }
};

struct D : A, B
{
  D() : B(static_cast<A*>(this), this)
  {}
};

struct Base
{
  virtual ~Base()
  {}
};

struct Derived : Base
{
  virtual void name()
  {}
};

int main()
{
  D d; // the most derived object
  A& a = d; // upcast, dynamic_cast maybe used, but unnecessary
  D& new_d = dynamic_cast<D&>(a); // downcast
  B& new_b = dynamic_cast<B&>(a); // sidecast

  Base* b1 = new Base;
  if (Derived* d = dynamic_cast<Derived*>(b1))
  {
    std::cout << "downcast from b1 to d successful \n";
    d->name(); // safe to call
  }

  Base* b2 = new Derived;
  if (Derived* d = dynamic_cast<Derived*>(b2))
  {
    std::cout << "downcast from b2 to d successful \n";
    d->name(); // safe to call
  }

  delete b1;
  delete b2;
}