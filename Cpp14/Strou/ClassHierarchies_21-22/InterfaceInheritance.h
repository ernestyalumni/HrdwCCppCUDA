//------------------------------------------------------------------------------
/// \file InterfaceInheritance.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Interface inheritance as classes only.
/// \ref Ch. 21 Class Hierarchies, 21.2.Design of Class Hierarchies
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details
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
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 Tuple2_main.cpp -o Tuple2_main
//------------------------------------------------------------------------------
#ifndef _CLASS_HIERARCHY_INTERFACE_INHERITANCE_H_
#define _CLASS_HIERARCHY_INTERFACE_INHERITANCE_H_

#include <ostream>

namespace ClassHierarchy
{

//------------------------------------------------------------------------------
/// \class Interface1
/// \brief Represents Ival_box class in Stroustrup examples
/// \ref Ch. 21 Class Hierarchies, 21.2.2 Interface Inheritance, pp. 617
/// \details Should contain no data.
//------------------------------------------------------------------------------
class Interface1
{
  public:

    // Gone is the ctor, since there's no data to initialize.

    virtual int value1() = 0;

    virtual void set_value1(const int i) = 0;

    virtual void run1() = 0;

    // Added Virtual destructor to ensure proper cleanup of data
    virtual ~Interface1()
    {}
}; // class Interface1

//------------------------------------------------------------------------------
/// \class Interface2
/// \brief Represents BBwidget or BBWindow class in Stroustrup's examples
/// \ref Ch. 21 Class Hierarchies, 21.2.2 Interface Inheritance, pp .617
/// \details Should contain no data.
//------------------------------------------------------------------------------
class Interface2
{
  public:

    virtual int value2() = 0;
    virtual void set_value2(const int i) = 0;
    virtual void run2() = 0;

    // error: templates may not be 'virtual'
    //template <typename T>
    //virtual T t() = 0;

    // error: templates may not be 'virtual'
    //template <typename T>
    //void set_t(const T& t) = 0; 

    virtual ~Interface2()
    {}
}; // class Interface2


//------------------------------------------------------------------------------
/// \class A
/// \brief Represents class A in dynamic_cast operator example
/// \ref https://www.bogotobogo.com/cplusplus/dynamic_cast.php
/// \details Should contain no data.
//------------------------------------------------------------------------------
class A
{
  public:

    virtual void f();

    virtual ~A()
    {}
}; // class A

class A1
{
  public:

    A1();
};

//------------------------------------------------------------------------------
/// \class Implementation1
/// \brief Represents Ival_slider class in Stroustrup examples
/// \ref Ch. 21 Class Hierarchies, 21.2.2 Interface Inheritance, pp. 617
//------------------------------------------------------------------------------
class Implementation1 : public Interface1
{
  public:

    Implementation1();
    Implementation1(const int value);

    int value1() override;

    void set_value1(const int i) override;

    void run1() override;

    // Added Virtual destructor to ensure proper cleanup of data
    ~Implementation1() override;    

  private:

    int value_;
}; // class Implementation1

//------------------------------------------------------------------------------
/// \class B
/// \brief Represents class B in dynamic_cast operator example
/// \ref https://www.bogotobogo.com/cplusplus/dynamic_cast.php
//------------------------------------------------------------------------------
class B : public A
{
  public:

    void f();
};

class B1 : public A, protected A1
{
  public:

    B1();
};


//------------------------------------------------------------------------------
/// \class IoObj
/// \brief Represents Io_obj class in Stroustrup examples.
/// \details Base class for object I/O system
/// \ref Ch. 22 RTTI, 22.2.1 `dynamic_cast`, pp. 644
//------------------------------------------------------------------------------
class IoObj
{
  public:

    virtual IoObj* clone() = 0;
}; // class IoObj

class IoDate : public A, public IoObj
{
  public:

    IoObj* clone();
};



} // namespace ClassHierarchy

#endif // _CLASS_HIERARCHY_INTERFACE_INHERITANCE_H_