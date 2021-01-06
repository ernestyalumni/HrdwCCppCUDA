//------------------------------------------------------------------------------
/// \file LoadPeakProblem.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Load the 2-dim. problem for peak finding of Problem Set 1, MIT 6.006.
/// \details MIT OCW 6.006, Fall 2011, Problem Set 1.
//------------------------------------------------------------------------------
#ifndef ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
#define ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H

#include <cstddef> // std::size_t
#include <fstream> // std::ifstream
#include <optional>
#include <string>
#include <utility> // std::pair
#include <vector>

namespace Algorithms
{
namespace PeakFinding
{

class LoadPeakProblem
{
  public:

    LoadPeakProblem();

    bool first_equal_sign_found() const
    {
    	return first_equal_sign_found_;
    }

    bool first_left_bracket_found() const
    {
    	return first_left_bracket_found_;
    }

    bool last_right_bracket_found() const
    {
    	return last_right_bracket_found_;
    }

    bool left_row_bracket_found() const
    {
    	return left_row_bracket_found_;
    }

    void parse(std::ifstream& input_file_stream);

    std::size_t number_of_rows() const
    {
    	return data_.size();
    }

    std::size_t ith_row_size(const std::size_t i) const
    {
    	return data_.at(i).size();
    }

    auto get(const std::size_t i, const std::size_t j) const
    {
    	return data_.at(i).at(j);
    }

  protected:

    //parse_equal_sign

  	std::optional<std::pair<size_t, size_t>> get_row_boundaries(const std::string& input_line);

  	std::vector<std::string> simple_parse_row(
  		const std::string& input_line,
  		const std::pair<size_t, size_t>& row_boundaries);

  	std::optional<std::size_t> parse_first_equal_sign(const std::string& input_line);
  	std::optional<std::size_t> parse_first_left_bracket(const std::string& input_line);
  	std::pair<std::optional<std::size_t>, std::optional<std::size_t>> parse_row_brackets(
  		const std::string& input_line);

  private:

  	bool first_equal_sign_found_ {false};
    bool first_left_bracket_found_{false};
    bool last_right_bracket_found_{false};
    bool left_row_bracket_found_{false};

    std::vector<std::vector<int>> data_;
};

} // namespace PeakFinding
} // namespace Algorithms

#endif // ALGORITHMS_PEAK_FINDING_LOAD_PEAK_PROBLEM_H
