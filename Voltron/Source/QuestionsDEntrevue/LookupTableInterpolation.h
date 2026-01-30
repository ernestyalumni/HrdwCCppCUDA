#ifndef QUESTIONS_D_ENTREVUE_LOOKUP_TABLE_INTERPOLATION_H
#define QUESTIONS_D_ENTREVUE_LOOKUP_TABLE_INTERPOLATION_H

#include <algorithm> // std::lower_bound
#include <concepts>
#include <type_traits>
#include <vector>

namespace QuestionsDEntrevue
{

//------------------------------------------------------------------------------
/// Returns interpolated action for given target velocity.
/// Clamps to nearest action if target is outside table range.
/// Assumes velocities is non-empty and strictly sorted ascending.
/// velocities server as the lookup table for actions, lookup table in this
/// example for interpolation.
//------------------------------------------------------------------------------
template <std::floating_point T>
T interpolate_action(
  const std::vector<T>& velocities,
  const std::vector<T>& actions,
  T target_velocity)
{
  if (velocities.empty() || velocities.size() != actions.size())
  {
    throw std::invalid_argument("Invalid or mismatched lookup tables");
  }

  // Binary search: finds first position where velocities [i] >= target
  auto it = std::lower_bound(
    velocities.begin(),
    velocities.end(),
    target_velocity);

  // Case 1: target <= first entry -> clamp to first action
  if (it == velocities.begin())
  {
    return actions[0];
  }

  // Case 2: target > last_entry -> clamp to last action
  if (it == velocities.end())
  {
    return actions.back();
  }

  // Case 3: target is between it - 1 and it
  const size_t high_index {static_cast<size_t>(it - velocities.begin())};
  const size_t low_index {high_index - 1};

  const T v_low {velocities[low_index]};
  const T v_high {velocities[high_index]};
  const T a_low {actions[low_index]};
  const T a_high {actions[high_index]};

  // Avoid divide by zero (shouldn't happen if strictly increasing, but
  // defensive)
  if (v_high == v_low)
  {
    return a_low;
  }

  const T fraction {(target_velocity - v_low) / (v_high - v_low)};
  return a_low + (a_high - a_low) * fraction;
}

} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_LOOKUP_TABLE_INTERPOLATION_H