//------------------------------------------------------------------------------
/// \file PassionneDordinateurRang.h
/// \author Ernest Yeung
/// \brief
/// \ref 
//------------------------------------------------------------------------------
#ifndef QUESTIONS_DENTREVUE_PASSIONNE_DORDINATEUR_RANG_H
#define QUESTIONS_DENTREVUE_PASSIONNE_DORDINATEUR_RANG_H

#include <string>
#include <vector>

namespace QuestionsDEntrevue
{

namespace PassionneDordinateurRang
{

namespace ActivezLesFontaines
{

int activation_de_fontaine(std::vector<int> emplacements);

//------------------------------------------------------------------------------
/// \details Complexity: Time complexity O(N).
/// Keep in mind to copy results is another O(N).
//------------------------------------------------------------------------------
std::vector<int> compute_left_ranges(std::vector<int>& emplacements);

//------------------------------------------------------------------------------
/// \details Complexity: Time complexity O(N).
/// Keep in mind to copy results is another O(N).
//------------------------------------------------------------------------------
std::vector<int> compute_right_ranges(std::vector<int>& emplacements);

std::vector<int> count_fountains_by_memo(std::vector<int>& emplacements); 

std::string couper_gauche(const std::string&);

} // namespace ActivezLesFontaines

} // namespace PassionneDordinateurRang

} // namespace QuestionsDEntrevue

#endif // QUESTIONS_DENTREVUE_PASSIONNE_DORDINATEUR_RANG_H