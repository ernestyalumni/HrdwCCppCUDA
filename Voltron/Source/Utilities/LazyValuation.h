//------------------------------------------------------------------------------
/// \file LazyValuation.h
/// \author Ernest Yeung
/// \brief .
/// \ref https://gitlab.com/manning-fpcpp-book/code-examples/blob/master/chapter-06/lazy-val/main.cpp
///-----------------------------------------------------------------------------
#ifndef _UTILITIES_LAZY_VALUATION_H_
#define _UTILITIES_LAZY_VALUATION_H_

#include <functional>
#include <string>
#include <iostream>
#include <mutex>
#include <optional>

namespace Utilities
{

//------------------------------------------------------------------------------
/// \class LazyValuation
///
/// \brief Store a computation and cache result of computation after it's
/// executed (called memoization)
/// \ref 6.1 Laziness in C++ pp. 123, Functional Programming in C++
/// \details This type needs to hold the following:
/// * the computation
/// * flag indicating whether you've already calculated the result
/// * the calculated result
///
/// Make LazyValuation template class immutable, at least when looked at from
/// the outside world. You'll mark all member functions created as const.
/// internally, need to be able to change cached value after first calculating
/// it, so all cache-related member variables must be declared as mutable.
//------------------------------------------------------------------------------

// Choice to have computation type as template parameter for LazyValuation
// results in:
// Because defining computatoina through lambdas, won't be able to specify type
// explicitly. Need to create make_lazy_valuation function because automatic
// template argument deduction works for function templates for C++14 and
// earlier: C++17 supports automatic template deduction for types.
template <typename F>
class LazyValuation
{
	private:

		// Stores the function object that defines the computation.
		F computation_;

		// In the book, we are using a value and a Boolean flag to denote whether we
		// have already calculated the value or not. In this implementation, we are
		// using std::optional which is like a container that can either be empty or
		// it can contain a single value exactly what we tried to simulate with the
		// value/Boolean flag pair. Optional values will be covered in more detail
		// in Cukic's Ch. 9.

		mutable std::optional<decltype(computation_())> value_;

		// You need the mutex in order to stop multiple threads from trying to
		// initialize the cache at the same time.
		mutable std::mutex value_lock_;

	public:

		//--------------------------------------------------------------------------
		/// \fn Constructor
		/// \brief Constructor
		/// \details Constructor doesn't need to do anything except store the
		/// computation and set to false the flag that indicates whether you've
		/// already calculated the result.
		//--------------------------------------------------------------------------

		LazyValuation(F function) : computation_{function}
	{}

		LazyValuation(LazyValuation&& other) :
			computation_{std::move(other.computation_)}
		{}

	// Allows implict casting of an instance of LazyValuation to the const-ref of
	// the return type of the computation.
	operator decltype(computation_())() const
	{
		// std::unique_lock<std::mutex> lock {cache_mutex_};
		// This is suboptimal: every time program needs the value, you lock and
		// unlock a mutex. But you need to lock the cache_ variable only the 1st.
		// time the function is called - while you're calculating the value.
		// cf. Listing 6.3, pp. 125, Ch. 6 Lazy evaluation of Cukic

		// Forbids concurrent access to the cache
		std::lock_guard<std::mutex> lock(value_lock_);

		// Caches the result of the computation for later use.

		if (!value_)
		{
			value_ = std::invoke(computation_);
		}

		return *value_;
	}
};

template <typename F>
LazyValuation<F> make_lazy_valuation(F&& function)
{
	return LazyValuation<F>(std::forward<F>(function));
}

// Instead of the make_lazy_valuation helper function, we can define the 'lazy'
// keyword with a bit of macro trickery - the macro will call the operator minus
// and create the lambda head - we can only provide the lambda body when
// creating the lazy value

struct _MakeLazyValuationHelper
{
	template <typename F>
	auto operator-(F&& function) const
	{
		return LazyValuation<F>(function);
	}
} _MakeLazyValuationHelper;

#define lazy _MakeLazyValuationHelper - [=]


} // namespace Utilities

#endif // _UTILITIES_LAZY_VALUATION_H_