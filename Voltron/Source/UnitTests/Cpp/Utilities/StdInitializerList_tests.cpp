#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <initializer_list>
#include <iterator> // std::back_inserter
#include <vector>

using std::vector;

template <typename T>
class ProcessInitializerList : public vector<T>
{
	public:

		ProcessInitializerList():
			vector<T>{},	
			size_{0},
			constructed_with_list_{false}
		{}

		ProcessInitializerList(const std::initializer_list<T> list):
			vector<T>{},
			size_{list.size()},
			constructed_with_list_{true}
		{
			// cf. https://en.cppreference.com/w/cpp/algorithm/copy
			// Complexity, exactly (last - first) assignments.
			// In practice, std::copy implementations avoid multiple assignments and
			// use bulk copy functions such as std::memmove when value type is
			// trivially copyable.

			//------------------------------------------------------------------------
			/// \details
			/// template <class Container>
			/// std::back_insert_iterator<Container> back_inserter(Container& c)
			/// Returns std::back_insert_iterator which can be used to add elements to
			/// end of container c.
			///
			/// The reason why you need to use std::back_inserter with std::vector:
			///
			/// cf. https://stackoverflow.com/questions/17104227/copy-algorithm-with-back-inserter
			/// A push_back makes vector reallocate and move data somewhere else in
			/// the memory, then next element copied will have a wrong address for
			/// source, which ends up with segmentation fault.
			//------------------------------------------------------------------------

			std::copy(list.begin(), list.end(), std::back_inserter(*this));

			// This works, but try std::copy for efficiency
			//for (const auto x : list)
			//{
			//	this->emplace_back(x);
			//}
		}

		bool constructed_with_list() const
		{
			return constructed_with_list_;
		}

	private:

		std::size_t size_;
		bool constructed_with_list_;
};

BOOST_AUTO_TEST_SUITE(Cpp)
BOOST_AUTO_TEST_SUITE(Utilities)
BOOST_AUTO_TEST_SUITE(StdInitializerList_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructs)
{
	ProcessInitializerList<double> list;

	BOOST_TEST(!list.constructed_with_list());
	BOOST_TEST(list.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithEmptyBraceEnclosedList)
{
	ProcessInitializerList<double> list {};

	BOOST_TEST(!list.constructed_with_list());
	BOOST_TEST(list.empty());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithNonEmptyBraceEnclosedList)
{
	ProcessInitializerList<int> list {42, 43, 44};

	BOOST_TEST(list.constructed_with_list());
	BOOST_TEST(!list.empty());

	for (int i {0}; i < 3; ++i)
	{
		BOOST_TEST(list[i] = 42 + i);
	}
}

BOOST_AUTO_TEST_SUITE_END() // StdInitializerList_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp