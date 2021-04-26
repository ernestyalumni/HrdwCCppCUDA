#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef> // std::size_t
#include <initializer_list>
#include <vector>

using std::vector;

template <typename T>
class ProcessInitializerList : public vector<T>
{
	public:

		ProcessInitializerList():
			vector<T>{},	

			constructed_with_list_{false}
		{}

		ProcessInitializerList(const std::initializer_list<T> list):
			vector<T>{},
			constructed_with_list_{true}
		{
			std::copy(list.begin(), list.end(), this->begin());
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


}

BOOST_AUTO_TEST_SUITE_END() // StdInitializerList_tests
BOOST_AUTO_TEST_SUITE_END() // Utilities
BOOST_AUTO_TEST_SUITE_END() // Cpp