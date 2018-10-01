//------------------------------------------------------------------------------
/// \file enable_if.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate enable_if metafunction for template metaprogramming..
/// \ref https://en.cppreference.com/w/cpp/types/enable_if
/// \details This metafunction is a convenient way to leverage SFINAE to
/// conditionally remove functions from overload resolution, based on type
/// traits, and to provide separate function overloads and specializations for
/// different type traits.
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
///  g++ -std=c++14 virtual_dtors_main.cpp -o virtual_dtors_main
//------------------------------------------------------------------------------
#ifndef _TEMPLATES_ENABLE_IF_H_
#define _TEMPLATES_ENABLE_IF_H_

#include <iostream>
#include <string>
#include <type_traits>

namespace Templates
{

namespace Details
{

struct inplace_t{};

struct MakeNew
{
  void* operator()(std::size_t sz, void* p, inplace_t)
  {
    return p;
  } 

  void* operator new(std::size_t, void* p, inplace_t)
  {
    return p;
  }

};

} // namespace Details

// \ref https://stackoverflow.com/questions/6210921/operator-new-inside-namespace
// An allocation function shall be a class member function or global function;
// a program is ill-formed if allocation function declared in namespace scope.
//void* operator new(std::size_t, void* p, Details::inplace_t)
//void* new_guy(std::size_t, void* p, Details::inplace_t)
//{
//  return p;
//}


// #1, enabled via the return type
template <class T, class... Args>
typename std::enable_if_t<std::is_trivially_constructible<T, Args&&...>::value>
  construct(T* t, Args&&...args)
{
  std::cout << "constructing trivially constructible T\n";
}

// #2 
template <class T, class... Args>
// Using helper type
std::enable_if_t<!std::is_trivially_constructible<T, Args&&...>::value> 
construct(T* t, Args&&... args)
{
  std::cout << "constructing non-trivially constructible T\n";
  // Segmentation Fault
//  Details::MakeNew::operator new(1, t, Details::inplace_t{}) T(args...);
//  Details::MakeNew::operator new(1, t, Details::inplace_t{});

}

// #3, enabled via a parameter
template <class T>
void destroy(
  T* t,
  typename std::enable_if_t<std::is_trivially_destructible<T>::value>* = 0
  )
{
  std::cout << "destroying trivially destructible T\n";
}

// #4, enabled via a template parameter
template <class T,
  typename std::enable_if_t<
    !std::is_trivially_destructible<T>{} &&
    (std::is_class<T>{} || std::is_union<T>{}),
    int> = 0>
void destroy(T* t)
{
  std::cout << "destroying non-trivially destructible T\n";
  t->~T();
}

// #5, enabled via a template parameter
template <class T,
  typename = std::enable_if_t<std::is_array<T>::value>
  >
void destroy(T* t) // note, function signature is unmodified
{
  for (std::size_t i {0}; i < std::extent<T>::value; ++i)
  {
    destroy((*t)[i]);
  }
}


//template <class T,
//  typename = std::enable_if_t<std::is_void<T>::value>
//  >
//void destroy(T* t){} // error: has the same signature with #5

// the partial specialization of A is enabled via a template parameter

// the partial specialization of A is enabled via a template parameter
template <class T, class Enable = void>
class A{}; // primary template

template <class T>
class A<T, typename std::enable_if_t<std::is_floating_point<T>::value>>
{}; // specialization for floating point types

} // namespace Templates

#endif // _TEMPLATES_ENABLE_IF_H_
