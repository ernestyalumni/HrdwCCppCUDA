//------------------------------------------------------------------------------
// \file StdForward_tests.cpp
//------------------------------------------------------------------------------

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility> // std::forward

BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(StdForward_tests)

BOOST_AUTO_TEST_SUITE(CppReferenceExamples)

struct A
{
	A(int&& n)
	{
		std::cout << "rvalue overload, n=" << n << "\n";
	}	

	A(int& n)
	{
		std::cout << "lvalue overload, n=" << n << "\n";
	}
};

class B
{
	public:

		template <class T1, class T2, class T3>
		B(T1&& t1, T2&& t2, T3&& t3) :
			a1_{std::forward<T1>(t1)},
			a2_{std::forward<T2>(t2)},
			a3_{std::forward<T3>(t3)}
		{}

	private:

		A a1_, a2_, a3_;
};

template <class T, class... U>
std::unique_ptr<T> make_unique2(U&&... u)
{
  return std::unique_ptr<T>(new T(std::forward<U>(u)...));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstratePerfectForwardingOfParametersAndParameterPacks)
{
  int i = 1;

  auto t = make_unique2<B>(2, i, 3);
  
  BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // CppReferenceExamples

// cf. https://github.com/SuperV1234/vittorioromeo.info/blob/master/extra/cpp17_curry/fwd_capture.hpp
//
// #define FWD(...) std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
// e.g. FWD(f)()

template <typename Function>
constexpr decltype(auto) forward_a_function(Function&& f)
{
	// If f() can be called, then immediately call and return.
	// This is the "Base case."

	//if constexpr (std::is_invokable<Function&&()>{})

	return std::forward<decltype(f)>(f);
}

BOOST_AUTO_TEST_SUITE(VittorioRomeoExamples)


BOOST_AUTO_TEST_SUITE_END() // VittorioRomeoExamples

// cf. https://en.cppreference.com/w/cpp/types/is_invocable

auto func2(char) -> int(*)()
{
	return nullptr;
}

int dummy_f0()
{ return 42; }

void dummy_f0b()
{ return; }

int dummy_f1(const int x)
{ return x; }

void dummy_f1b(const int x)
{ 
	std::cout << x << '\n';	
	return;
}


/*
template <
	typename Function,
	typename std::enable_if<std::is_invocable_v<Function>>::type = 0
	>
*/

//template <typename Function>
//bool is_invocable_t()
//{
//	return std::is_invocable_v<Function, Arg>
//}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(IsInvocableWorks)
{
	BOOST_TEST((std::is_invocable<int()>::value));
	BOOST_TEST((std::is_invocable_r<int, int()>::value));
	BOOST_TEST((std::is_invocable_r<void, void(int), int>::value));
	BOOST_TEST((std::is_invocable_r<int(*)(), decltype(func2), char>::value));
}

BOOST_AUTO_TEST_CASE(IsInvocableDistinguishesBetween0And1Arguments)
{
	BOOST_TEST((std::is_invocable_v<decltype(dummy_f0)>));
	BOOST_TEST(!(std::is_invocable_v<decltype(dummy_f1)>));
	BOOST_TEST((std::is_invocable_v<decltype(dummy_f0b)>));
	BOOST_TEST(!(std::is_invocable_v<decltype(dummy_f1b)>));


}



BOOST_AUTO_TEST_SUITE_END() // StdForward_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities