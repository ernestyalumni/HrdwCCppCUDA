#ifndef ALGORITHMS_EXPERT_IO_LEVEL_2_H
#define ALGORITHMS_EXPERT_IO_LEVEL_2_H

#include <vector>

namespace Algorithms
{
namespace ExpertIo
{

namespace MergeOverlappingIntervals
{

//------------------------------------------------------------------------------
/// \url https://www.algoexpert.io/questions/Merge%20Overlapping%20Intervals
/// \brief Merge Overlapping Intervals, AlgoExpert, Difficulty Medium
///
/// \details Sort the intervals with respect to their starting values.
//------------------------------------------------------------------------------
std::vector<std::vector<int>> merge_overlapping_intervals(
  std::vector<std::vector<int>> intervals);

void insertion_sort_intervals(std::vector<std::vector<int>>& input // [in, out]
  );

} // namespace MergeOverlappingIntervals

} // namespace ExpertIo
} // namespace Algorithms

#endif // ALGORITHMS_EXPERT_IO_LEVEL_2_H