//------------------------------------------------------------------------------
/// \file   : stdinitializer_list_eg.cpp
/// \author : Ernest Yeung
/// \email  : ernestyalumni@gmail.com
/// \brief  : Example of std::initializer_list.  
/// \details Object of type std::initializer_list<T>, lightweight proxy object 
/// 	that provides access to an array of objects of type const T
/// \ref    : http://en.cppreference.com/w/cpp/utility/initializer_list
/// Ch. 17 Constructors; Bjarne Stroustrup, 17.3.4 Initializer-List
/// Constructors. The C++ Programming Language, 4th Ed., Stroustrup; 
/// If you find this code useful, feel free to donate directly and easily at
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
/// 
/// COMPILATION TIPS:
///  g++ -std=c++14 stdinitializer_list_eg.cpp -o stdinitializer_list_eg
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <initializer_list>

#include <algorithm> // std::copy
#include <vector>

template <class T>
struct S
{
	std::vector<T> v;
	S(std::initializer_list<T> l):
		v{l}
	{
		std::cout << "constructed with a " << l.size() << "-element list\n";
	}

	void append(std::initializer_list<T> l)
	{
		v.insert(v.end(), l.begin(), l.end());
	}

	std::pair<const T*, std::size_t> c_arr() const
	{
		return {&v[0], v.size()}; // copy list-initialization in return statement
															// this is NOT a use of std::initializer_list
	}
};

template <typename T>
void templated_fn(T) 
{}

/// \ref pp. 498 Ch. 17 Construction, Stroustrup
/// \details elements of initializer_list are immutable.
int f(std::initializer_list<int> x, int val)
{
  // *x.begin() = val; // error: attempt to change the value of an
  // initializer-list element
  return *x.begin(); // OK
}

void g()
{
  for (int i {0}; i != 10; ++i)
  {
    std::cout << f({1, 2, 3}, i) << '\n';
  }
}


int main()
{
	S<int> s = {1, 2, 3, 4, 5}; // copy list-initialization
  S<int> s2 {6, 7, 8, 9};   

  std::cout << "The vector size is now " << s.c_arr().second << " ints:\n";

  std::cout << "The s2 size is now " << s2.c_arr().second << " ints:\n";

  s.append({6, 7, 8});    // list-initialization in function call
  s2.append({6, 7});    // list-initialization in function call

  std::cout << "The vector size is now " << s.c_arr().second << " ints:\n";

  std::cout << "The s2 size is now " << s2.c_arr().second << " ints:\n";

  for (auto n : s.v)
  {
    std::cout << n << ' ';
  }
  std::cout << '\n';

  for (auto n : s2.v)
  {
    std::cout << n << ' ';
  }
  std::cout << '\n';

  std::cout << "Range for over brace-init-list: \n";

  for (int x : {-1, -2, -3})  // the rule for auto makes this ranged-for work
  {
    std::cout << x << ' ';
  }
  std::cout << '\n';

  auto al = {10, 11, 12};     // special rule for auto
//  auto al2 {13, 14, 15}; // error direct-list-initialization of auto requires
    // exactly 1 element  

  std::cout << "The list bound to auto has size() = " << al.size() << '\n';

//  std::cout << "The 2nd list bound to auto has size() = " << al2.size() << '\n';

  // templated_fn({1, 2, 3}); // compiler error! "{1, 2, 3}" is not an expression,
    // it has no type, and so T cannot be deduced.

//  templated_fn({1,2,3}); // error no matching function for call to 
  // templated_fn(<brace-enclosed initializer list)

  templated_fn<std::initializer_list<int>>({1, 2, 3});  // OK
  templated_fn<std::vector<int>>({1, 2, 3});            // also OK

  std::cout << "\n\n Now testing out std::initializer_list itself : \n";

  std::initializer_list<double> example_list {1., 2., 3., 4.};

  std::cout << " example_list.begin() " << *example_list.begin() << '\n';

  std::cout << *(example_list.begin()+1) << '\n';

  std::cout << *(example_list.end()-2) << '\n';

  std::cout << " example_list.size() : " << example_list.size() << '\n';

  // double*, std::initializer_list -> double* s.t. double* filled up with 
  double* ptr_to_doubles = const_cast<double*>(example_list.begin());
//  const_cast<double*>(example_list.begin()); // WORKS
//  const_cast<double*>(example_list); // DOES NOT WORKS

  for (int i {0}; i < example_list.size(); i++)
  {
    std::cout << ptr_to_doubles[i] << " ";
  } std::cout << std::endl;

  for (auto iter {example_list.begin()}; iter != example_list.end(); iter++)
  {
    std::cout << *iter << " ";
  } std::cout << std::endl;

  for (auto x : example_list)
  {
    std::cout << x << " ";
  } std::cout << std::endl;

  double* ptr_to_5_elements {new double[5]};
  for (int i {1}; i < 6; i++)
  {
    ptr_to_5_elements[i-1] = i;
  }
  for (int i {0}; i < 5; i++)
  {
    std::cout << ptr_to_5_elements[i] << " ";
  } std::cout << std::endl;

  double* ptr_to_6_elements {new double[6]};
  for (int i {0}; i < 5; i++)
  {
    ptr_to_6_elements[i] = ptr_to_5_elements[i];
    std::cout << ptr_to_6_elements[i] << " ";
  } std::cout << std::endl;

  //----------------------------------------------------------------------------
  /// \details Let's std::copy an std::initializer_list into a raw pointer
  //----------------------------------------------------------------------------
  std::initializer_list<double> example_list2 {11., 22., 33., 44., 55.};
  std::copy(example_list2.begin(), example_list2.end(), &ptr_to_5_elements[0]);
  for (int i {0}; i < 5; i++)
  {
    std::cout << ptr_to_5_elements[i] << ' ';
  } 

  std::cout << std::endl;

  delete[] ptr_to_5_elements;
  delete[] ptr_to_6_elements;

  std::cout << " f : " << '\n';

  f({1,2,3,4,5,6,7}, 9);
  g();

  /// \ref pp. 498 17.3.4.3. Direct and Copy Initialization
  /// \details distinction between direct initialization and copy
  /// initialization (Sec. 16.2.6) maintained for `{}` initialization. For a
  /// container, this implies that distinction is applied to both container and
  /// its elements
  /// * The container's initializer-list ctor can be explicit or not
  /// * ctor of element type of initializer list can be explicit or not
  /// direct initialization vs. copy initialization distinction applied to
  /// elements.
  std::vector<std::vector<double>> vs = 
  {
    {10, 11, 12, 13, 14}, // OK: vector of 5 elements
    {10}, // OK: vector of 1 element
    // 10, // error : vector<double>(int) is explicit
    std::vector<double>{10, 11, 12, 13}, // OK : vector for 5 elements
    std::vector<double>{10}, // OK vector of 1 element with value 10.0
    std::vector<double>(10), // OK: vector of 10 elements with value 0.0

  };

  std::vector<double> v1 {7}; // OK : v1 has 1 element (with the value 7)
  std::vector<double> v2 = {9}; // OK: v2 has 1 element (with value 9)

  std::vector<double> v12 {7, 8, 9}; // OK: v1 has 3 elements with values 
  // {7,8,9}
  std::vector<double> v22 = {9, 8, 7}; // OK: v2 has 3 elements with values 
  // { 9,8,7}

  std::vector<std::string> v13 { "Anya"}; // OK: v1 has 1 element 
  std::vector<std::string> v23 = {"Courtney"}; // OK; v2 has 1 element



}
