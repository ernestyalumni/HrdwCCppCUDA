//------------------------------------------------------------------------------
/// \file Mutable_tests.cpp
///
/// \ref Ch. 5 of Cukic.
//------------------------------------------------------------------------------
#include <boost/test/unit_test.hpp>

#include <iostream> // std::cerr
#include <list> // std::list
#include <numeric> // std::accumulate
#include <vector>

BOOST_AUTO_TEST_SUITE(Mutable)
BOOST_AUTO_TEST_SUITE(Mutable_tests)

BOOST_AUTO_TEST_SUITE(Mutable_tests)

// cf. pp. 101, Sec. 5.1. of Cukic
class Movie
{
  public:

    Movie(const std::string& name, const std::list<int>& scores):
      name_{name},
      scores_{scores}
    {}

    double average_score() const;

    void add_score_to_the_back(const int new_score)
    {
      // Appends new element to end of container; element is constructed through
      // std::allocator_traits::construct, which typically uses placement-new to
      // construct element in-place at location provided by container.
      scores_.emplace_back(new_score);
    }

  private:

    std::string name_;
    std::list<int> scores_;
};

double Movie::average_score() const
{
  // Calling being and end on a const value is the same as calling cbegin and
  // cend.
  return std::accumulate(scores_.begin(), scores_.end(), 0) /
    static_cast<double>(scores_.size());
}

// cf. 5.1 Problems with the mutable state, pp. 101
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(MutableTest)
{
  Movie movie {"Star Wars: Rise of the Skywalker", {9, 7, 10, 5}};

  BOOST_TEST(movie.average_score() == 7.75);
}

double max(const std::vector<double>& numbers)
{
  // Assume the numbers vector isn't empty, to have the std::max_element return
  // a valid iterator.
  assert(!numbers.empty());
  auto result = std::max_element(numbers.cbegin(), numbers.cend());

  // Writes numbers 1, 2, 3 to std::cerr. The max function isn't referentially
  // transparent, and therefore it isn't pure.
  std::cerr << "Maximum is: " << *result << std::endl;
  return *result;
}

double pure_max(const std::vector<double>& numbers)
{
  auto result = std::max_element(numbers.cbegin(), numbers.cend());
  return *result;
}

// cf. Listing 5.2 Search for and logging the maximum value, pp. 104
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ReferentialTransparencyExamples)
{
  auto sum_max = max({1}) + max({1, 2}) + max({1, 2, 3});

  BOOST_TEST(sum_max == 6);
}


BOOST_AUTO_TEST_SUITE_END() // Mutable_tests

BOOST_AUTO_TEST_SUITE_END() // Mutable_tests
BOOST_AUTO_TEST_SUITE_END() // Mutable
