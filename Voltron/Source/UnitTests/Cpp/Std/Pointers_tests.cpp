//------------------------------------------------------------------------------
/// \file Pointers_tests.cpp
/// \ref 
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

BOOST_AUTO_TEST_SUITE(Cpp) // The C++ Language
BOOST_AUTO_TEST_SUITE(Pointers_test)

std::optional<bool> is_Foo_constructed {std::nullopt};

struct Foo
{
  Foo()
  {
    std::cout << "Foo...\n";

    is_Foo_constructed = true;
  }

  ~Foo()
  {
    std::cout << "~Foo...\n";
    is_Foo_constructed = false;
  }
};

struct D
{
  void operator()(Foo* p)
  {
    std::cout << "Calling delete for Foo object...\n";
    delete p;
  }
};

std::optional<bool> is_Quantity_constructed {std::nullopt};

class Quantity
{
  public:
    Quantity() = delete;

    Quantity(const double x):
      x_{x}
    {
      is_Quantity_constructed = true;
    }

    ~Quantity()
    {
      is_Quantity_constructed = false;
    }

    double x() const
    {
      return x_;
    }

  private:
    double x_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstratePointers)
{
  {
    // For type T, T* is type "pointer to T", i.e.
    // variable of type T* can hold address of an object of type T
    char c = 'a';
    char* p = &c;

    BOOST_TEST(c == 'a');
    std::cout << " c : " << c << " p : " << p << " c as unsigned int : " <<
      static_cast<unsigned int>(c) << " *p : " << *p << " &p : " << &p <<
        " &c : " << &c <<  '\n';
    BOOST_TEST(*p == 'a');
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateNullPtr)
{
  {
    void* x {nullptr};

    &x;

    BOOST_TEST(true);

    x;

    BOOST_TEST(true);

    // error: 'void*' is not a pointer-to-object type
    //*x;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateUniquePointers)
{
  {
    // https://en.cppreference.com/w/cpp/memory/unique_ptr/reset

    // Creating new Foo...\n, D is a custom deleter
    std::cout << "Creating new Foo...\n";
    BOOST_TEST(!static_cast<bool>(is_Foo_constructed));

    std::unique_ptr<Foo, D> up {new Foo(), D()}; // up owns the Foo pointer
    // (deleter D)
    BOOST_TEST(static_cast<bool>(is_Foo_constructed));
    BOOST_TEST(is_Foo_constructed.value());

    // Replace owned Foo with a new Foo...\n
    std::cout << "Replace owned Foo with a new Foo...\n";
    up.reset(new Foo()); // calls deleter for the old one

    BOOST_TEST(static_cast<bool>(is_Foo_constructed));
    BOOST_TEST(!is_Foo_constructed.value());

    std::cout << "Release and delete the owned Foo...\n";

    up.reset(nullptr);

    BOOST_TEST(static_cast<bool>(is_Foo_constructed));
    BOOST_TEST(!is_Foo_constructed.value());
  }
  {
    std::unique_ptr<Quantity> u_ptr {std::make_unique<Quantity>(3)};

    BOOST_TEST(u_ptr->x() == 3.);

    Quantity q1 {5};

    // Releases the ownership of the managed object if any.
    u_ptr.release();
    
    // SIGABRT applicat abort requested
    //u_ptr.reset(&q1);

    // THIS WORKS
    u_ptr = std::make_unique<Quantity>(q1);

    BOOST_TEST(u_ptr->x() == 5.);

    Quantity q2 {8};

    is_Quantity_constructed = true;

    u_ptr = std::make_unique<Quantity>(q2);

    BOOST_TEST(u_ptr->x() == 8.);
    BOOST_TEST(q2.x() == 8.);

    // This refers to q1 going out of scope and being deleted.
    BOOST_TEST(is_Quantity_constructed.value() == false);
  }
  {
    Quantity q1 {42};
    std::unique_ptr<Quantity> u_ptr {nullptr};
    u_ptr = std::make_unique<Quantity>(q1);
    BOOST_TEST(u_ptr->x() == 42);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateUniquePointersInClasses)
{
  // TODO
}

class IntString
{
  public:

    IntString(const int x, const std::string& s):
      x_{x},
      s_{s}
    {}

    int x() const
    {
      return x_;
    }

    const std::string& s() const
    {
      return s_;
    }

    void x(const int x)
    {
      x_ = x;
    }

    void s(const std::string& s)
    {
      s_ = s;
    }

  private:

    int x_;
    std::string s_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateReleasingUniquePointers)
{
  {
    IntString int_string {42, "Peek"};
    const IntString const_int_string {43, "Linked List"};

    std::unique_ptr<IntString> u_ptr {std::make_unique<IntString>(int_string)};

    BOOST_TEST(u_ptr->x() == 42);
    BOOST_TEST(u_ptr->s() == "Peek");

    u_ptr.release();

    BOOST_TEST(int_string.x() == 42);
    BOOST_TEST(int_string.s() == "Peek");

    u_ptr = std::make_unique<IntString>(const_int_string);

    BOOST_TEST(u_ptr->x() == 43);
    BOOST_TEST(u_ptr->s() == "Linked List");

    BOOST_TEST(const_int_string.x() == 43);
    BOOST_TEST(const_int_string.s() == "Linked List");

    std::cout << " &u_ptr : " << &u_ptr << '\n';

    auto released_ptr = u_ptr.release();

    std::cout << " u_ptr release: " << &released_ptr << ' ' << released_ptr << '\n';

    u_ptr = std::make_unique<IntString>(int_string);

    BOOST_TEST(int_string.x() == 42);
    BOOST_TEST(int_string.s() == "Peek");

    int_string.x(69);
    int_string.s("Last-in First-Out");

    BOOST_TEST(int_string.x() == 69);
    BOOST_TEST(int_string.s() == "Last-in First-Out");

    BOOST_TEST(u_ptr->x() == 42);
    BOOST_TEST(u_ptr->s() == "Peek");

    std::cout << " &u_ptr : " << &u_ptr << '\n';

    auto released_ptr1 = u_ptr.release();

    std::cout << " u_ptr release1: " << &released_ptr1 << ' ' << released_ptr1 << '\n';

    u_ptr = std::make_unique<IntString>(std::move(int_string));

    BOOST_TEST(u_ptr->x() == 69);
    BOOST_TEST(u_ptr->s() == "Last-in First-Out");

    int_string.x(70);
    int_string.s("Queues");

    BOOST_TEST(int_string.x() == 70);
    BOOST_TEST(int_string.s() == "Queues");

    BOOST_TEST(u_ptr->x() == 69);
    BOOST_TEST(u_ptr->s() == "Last-in First-Out");

    u_ptr->x(70);
    u_ptr->s("Queues");

    BOOST_TEST(u_ptr->x() == 70);
    BOOST_TEST(u_ptr->s() == "Queues");

    std::cout << " &u_ptr : " << &u_ptr << '\n';

    auto released_ptr2 = u_ptr.release();

    std::cout << " u_ptr release2: " << &released_ptr2 << ' ' << released_ptr2 << '\n';

  }
}

BOOST_AUTO_TEST_SUITE_END() // Pointers_test
BOOST_AUTO_TEST_SUITE_END() // Cpp