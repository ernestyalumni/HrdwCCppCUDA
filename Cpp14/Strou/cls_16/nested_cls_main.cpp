//------------------------------------------------------------------------------
/// \file nested_cls_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Examples demonstrating nested classes.
/// \ref https://en.cppreference.com/w/cpp/language/nested_types
/// 16.2.13 Member Types Ch. 16 Classes; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 16
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------

#include <iostream>

int x, y; // globals

class Enclose
{
  public:

    struct Inner // nested class
    {
      void f(int i)
      {
//        x = i; // Error: can't write to non-static enclose::x without instance.
        int a = sizeof x; // OK in C++11: operand of sizeof is unevaluated,
                          // this use of the non-static enclose::x is allowed.
        s_ = i; // OK: can assign to the static enclose::s
        ::x = i; // OK: can assign to global x
        y = i; // OK can assign to global y
      }

      void g(Enclose* p, int i)
      {
        p->x = i; // OK: assign to enclose::x
      }
    };

  private:
    int x; // note: private members
    static int s_;
};

// Out of class definitions of members of a nested class appear in namespace
// of enclosing class:

struct Enclose2
{
  struct Inner
  {
    static int x_;
    void f(int i);
  };
};

int Enclose2::Inner::x_ = 1; // definition
void Enclose2::Inner::f(int i)
{} // definition

// Nested classes can be forward-declared and later defined, either within the
// same enclosing class body, or outside of it:

class Enclose3
{
  class Nested1; // forward declaration
  class Nested2; // forward declaration
  class Nested1 {}; // definition of nested class
};

class Enclose3::Nested2 {}; // definition of nested class.

// Nested class declarations obey member access specifiers, a private member
// class cannot be named outside the scope of enclosing class, although objects
// of that class maybe manipulated:

class Enclose4
{
  struct Nested;

  public:
    static Nested f()
    {
      return Nested {};
    }

  private:
    struct Nested     // private member
    {
      void g() {}
    };
};

/// Stroustrup, 16.2.13. Member Types. pp. 469.
// \details 

template <typename T>
class Tree
{
  class Node;

  public:
    void g(Node*);

  private:

    using value_type = T;       // member alias

    enum Policy   // member enum
    {
      rb,
      splay,
      treeps
    };    

    class Node
    {
      public:

        void f(Tree*);

      private:

        Node* right;
        Node* left;
        value_type value;
    };
};

template <typename T>
void Tree<T>::Node::f(Tree* p)
{
//  top = right; // error: no object of type Tree specified
  p->top = right; // OK
  value_type v = left->value; // OK: value_type is not associated with an
    // object
}

template <typename T>
void Tree<T>::g(Tree<T>::Node* p)
{
//  value_type val = right->value;  // error: no object of type Tree::Node
  //value_type v = p->right->value; // error: Node::right is private
  p->f(this);
}



int main()
{
  Enclose test_enclose;
  Enclose::Inner test_inner;

  // Enclose4::Nested n1 = Enclose4::f(); // error: 'nested' is private

  Enclose4::f().g(); // OK: does not name 'nested'
  auto n2 = Enclose4::f();  // OK: does not name 'nested'
  n2.g();

}
