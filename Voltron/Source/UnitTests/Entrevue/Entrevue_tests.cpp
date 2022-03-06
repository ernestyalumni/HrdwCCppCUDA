//------------------------------------------------------------------------------
/// \file Entrevue_tests.cpp
///
/// \brief Blank unit tests for doing interview questions on the fly.
//------------------------------------------------------------------------------
#include "QuestionsDEntrevue/PassionneDordinateurRang.h"

#include <algorithm> // std::max, std::min
#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>

using
	QuestionsDEntrevue::PassionneDordinateurRang::ActivezLesFontaines::
		compute_left_ranges;
using
	QuestionsDEntrevue::PassionneDordinateurRang::ActivezLesFontaines::
		compute_right_ranges;
using
	QuestionsDEntrevue::PassionneDordinateurRang::ActivezLesFontaines::
		count_fountains_by_memo;
using
	QuestionsDEntrevue::PassionneDordinateurRang::ActivezLesFontaines::
		couper_gauche;
using std::string;
using std::vector;

BOOST_AUTO_TEST_SUITE(Entrevue)
BOOST_AUTO_TEST_SUITE(Entrevue_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Demonstrate)
{
  //std::cout << "\n\n Entrevue tests \n\n";
  {
    BOOST_TEST(true);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(FinDemonstrate)
{
  //std::cout << "\n\n End Entrevue tests \n\n";
  {
    BOOST_TEST(true);
  }
}

BOOST_AUTO_TEST_SUITE_END() // Entrevue_tests

BOOST_AUTO_TEST_SUITE(PassionneDordinateurRang)

BOOST_AUTO_TEST_SUITE(ActivationDeFontaine_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CouperGaucheMarche)
{
	{
		const string problem {"  not this here "};
		string result {couper_gauche(problem)};
		BOOST_TEST(result[0] == 'n');
		BOOST_TEST(result[1] == 'o');
		BOOST_TEST(result[2] == 't');
		BOOST_TEST(result.substr(0, 3) == "not");
	}
}

vector<int> example_0 {1, 1, 1};
vector<int> example_1 {2, 0, 0, 0};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComputeLeftRangesWorks)
{
	{
		const auto result = compute_left_ranges(example_0);
		BOOST_TEST(result[0] == 1);
		BOOST_TEST(result[1] == 1);
		BOOST_TEST(result[2] == 2);
	}
	{
		const auto result = compute_left_ranges(example_1);
		BOOST_TEST(result[0] == 1);
		BOOST_TEST(result[1] == 2);
		BOOST_TEST(result[2] == 3);
		BOOST_TEST(result[3] == 4);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ComputeRightRangesWorks)
{
	{
		const auto result = compute_right_ranges(example_0);
		BOOST_TEST(result[0] == 2);
		BOOST_TEST(result[1] == 3);
		BOOST_TEST(result[2] == 3);
	}
	{
		const auto result = compute_right_ranges(example_1);
		BOOST_TEST(result[0] == 3);
		BOOST_TEST(result[1] == 2);
		BOOST_TEST(result[2] == 3);
		BOOST_TEST(result[3] == 4);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CountFountainsByMemoWorks)
{
	{
		const auto result = count_fountains_by_memo(example_0);
		BOOST_TEST(result.size() == 1);
		BOOST_TEST(result[0] == 1);
	}
	{
		const auto result = count_fountains_by_memo(example_1);
		BOOST_TEST(result.size() == 2);
		BOOST_TEST(result[0] == 0);
		BOOST_TEST(result[1] == 3);
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ActivezLesFontaines)
{


	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // ActivationDeFontaine_tests

BOOST_AUTO_TEST_SUITE_END() // PassionneDordinateurRang

BOOST_AUTO_TEST_SUITE_END() // Entrevue