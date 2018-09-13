//------------------------------------------------------------------------------
/// \file class_object_initialization_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for class object initialization examples..
/// \ref Ch. 17 Constructors; Bjarne Stroustrup, 17.3 Class Object
///   Initialization. The C++ Programming Language, 4th Ed., Stroustrup;
/// \details Initialize objects of a class with and without ctors.
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
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
///  g++ -std=c++14 class_object_initialization_main.cpp -o class_object_initialization_main
//------------------------------------------------------------------------------
#include <iostream>
#include <netinet/in.h> // ::sockaddr_in
#include <netinet/ip.h> // superset of previous
#include <string>

/// \details We can initialize objects of a class for which we haven't defined
/// ctor using
/// * memberwise initialization
/// * copy initialization, or
/// * default initialization (without an initializer or with an empty
///   initializer list)

struct Work 
{
  std::string author;
  std::string name;
  int year;
};

struct SocketAddressInContainer
{
  ::sockaddr_in SocketAddressIn;
};

struct Buf
{
  int count;
  char buf[16*1024];
};

void f()
{
  Buf buf1; // leave elements uninitialized
  Buf buf2 {};  // I really want to zero out those elements

  int* p1 = new int; // *p1 is uninitialized
  int* p2 = new int{}; // *p2 ==0 
  int* p3 = new int{7}; // *p3 == 7
}

/// \details If a ctor is declared for a class, some ctor will be used for
/// every object. It's an error to try to create an object without a proper
/// initializer as required by the ctors.
///
/// References and consts must be initialized (Sec. 7.7, Sec. 7.5). Therefore,
/// a class containing such members can't be default constructed unless the
/// programmer supplies in-class member initializers (Sec. 17.4.4), or defines
/// a default ctor that initializes them (Sec. 17.4.1)

int glob {9};

struct X
{
  const int a1 {7}; // OK
  const int& r {9}; // OK
  int& r1 {glob}; // OK
};



int main()
{
  /// \ref 17.3.1. Initialization Without Constructors, Stroustrup.
  /// \details We can't define a ctor for a built-in type, yet we can
  /// initialize it with a value of suitable type
  int a {1};
  char* p {nullptr};

  Work s9 {"Beethoven",
    "Symphony No. 9 in D minor, Op. 125; Choral",
    1824}; // memberwise initialization

  Work currently_playing {s9}; // copy initialization
  Work none {}; // default initialization

  SocketAddressInContainer socket_address_in_container;
  SocketAddressInContainer socket_address_in_container1 {
    8, // unsigned short = uint8_t
    808, // unsigned short = sa_family_t
    42, // struct in_addr = unsigned long = in_port_t 
    {0, 1, 0, 1, 1, 1, 0, 0} // char [8] 
  };

  // Where no ctor requiring arguments is declared, it's also possible to 
  // leave out the initializer completely.
  Work alpha;

  Buf buf0; // statically allocated, so initialized by default.

  f();

//  X x2 {2}; // OK
//  X x3 {x2}; // OK : a copy ctor is implicitly defined (Sec. 17.6)

  X x {};
  std::cout << " x : " << x.a1 << ' ' << x.r << ' ' << x.r1 << '\n';

  

}
