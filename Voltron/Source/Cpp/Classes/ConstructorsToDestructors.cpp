#include "ConstructorsToDestructors.h"

#include <cstddef> // std::size_T
#include <initializer_list>
#include <iostream> // std::clog
#include <stdexcept> // std::runtime_error
#include <string>
#include <utility>

using std::clog;
using std::initializer_list;
using std::size_t;
using std::string;
using std::to_string;

namespace Cpp
{
namespace Classes
{

//------------------------------------------------------------------------------
/// \ref  https://stackoverflow.com/questions/1563897/static-constant-string-class-member
/// This is for before C++17
//------------------------------------------------------------------------------
const string DefaultConstructs::default_ctor_message {
  "DefaultConstructs default constructs"};

const string DefaultConstructs::dtor_message {
  "DefaultConstructs destructs"};

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/9110487/undefined-reference-to-a-static-member
/// \details Linker doesn't know where to allocate and need to tell it manually.
/// Otherwise linker errors as code to initialize variable in multiple source
/// files (since so far it only knows header code).
//------------------------------------------------------------------------------

int DefaultConstructs::default_ctor_counter_;
int DefaultConstructs::dtor_counter_;

DefaultConstructs::DefaultConstructs():
  s_data_{},
  int_data_{},
  is_default_constructed_{true}
{
  default_ctor_counter_++; 

  clog << default_ctor_message << ":" << to_string(default_ctor_counter_);
}

DefaultConstructs::~DefaultConstructs()
{
  ++dtor_counter_;

  clog << dtor_message << ":" << to_string(dtor_counter_);
}

const string NoDefaultConstruction::ctor_1_message {
  "NoDefaultConstruction constructs #1"};

const string NoDefaultConstruction::dtor_message {
  "NoDefaultConstruction destructs"};

int NoDefaultConstruction::ctor_1_counter_;
int NoDefaultConstruction::dtor_counter_;

NoDefaultConstruction::NoDefaultConstruction(
  int& int_ref,
  const int data
  ):
  int_ref_{int_ref},
  int_data_{data}
{
  ++ctor_1_counter_;

  clog << ctor_1_message << ":" << to_string(ctor_1_counter_);
}

NoDefaultConstruction::~NoDefaultConstruction()
{
  ++dtor_counter_;

  clog << dtor_message << ":" << to_string(dtor_counter_);
}

NoDefaultConstruction return_no_default_construction_as_type(
  int& int_ref,
  const int data)
{
  NoDefaultConstruction temp_object {int_ref, data};

  return temp_object;
}

const string CopyConstructOnly::ctor_1_message {
  "CopyConstructOnly constructs #1"};

const string CopyConstructOnly::copy_ctor_message {
  "CopyConstructOnly copy constructs"};

const string CopyConstructOnly::dtor_message {"CopyConstructOnly destructs"};

int CopyConstructOnly::ctor_1_counter_;
int CopyConstructOnly::copy_ctor_counter_;
int CopyConstructOnly::dtor_counter_;

CopyConstructOnly::CopyConstructOnly(
  const int value,
  const size_t N,
  initializer_list<int> int_list
  ):
  int_data_{value},
  N_{N},
  array_{new int[N_]}
{
  ctor_1_counter_++; 

  clog << ctor_1_message << ":" << to_string(ctor_1_counter_);
}

CopyConstructOnly::CopyConstructOnly(const CopyConstructOnly& other):
  int_data_{other.int_data_},
  N_{other.N_},
  array_{new int[N_]}
{
  for (int index {0}; index < N_; ++index)
  {
    array_[index] = other.array_[index];
  }

  ++copy_ctor_counter_;

  clog << copy_ctor_message << ":" << to_string(copy_ctor_counter_);
}

CopyConstructOnly::~CopyConstructOnly()
{
  delete[] array_;

  clog << dtor_message << ":" << to_string(dtor_counter_);

  ++dtor_counter_;
}

const string MoveOnlyLight::ctor_message {"MoveOnlyLight constructs"};
const string MoveOnlyLight::move_ctor_message {"MoveOnlyLight move constructs"};
const string MoveOnlyLight::move_assign_message {
  "MoveOnlyLight move assign"};
const string MoveOnlyLight::dtor_message {"MoveOnlyLight destructs"};

int MoveOnlyLight::ctor_counter_;
int MoveOnlyLight::move_ctor_counter_;
int MoveOnlyLight::move_assign_counter_;
int MoveOnlyLight::dtor_counter_;

MoveOnlyLight::MoveOnlyLight(const int value):
  data_{value}
{
  if (value < 1)
  {
    throw std::runtime_error(
      "Runtime Error; MoveOnlyLight ctor: input value: " +
      to_string(value) +
      "less than 1 and thurs invalid.");
  }

  ++ctor_counter_; 

  clog << ctor_message << ":" << to_string(ctor_counter_);
}

MoveOnlyLight::MoveOnlyLight(MoveOnlyLight&& other):
  data_{other.data_}
{
  other.data_ = -1;

  ++move_ctor_counter_; 

  clog <<
    move_ctor_message <<
    ":" <<
    to_string(move_ctor_counter_) <<
    "data_:" <<
    to_string(data_) <<
    "other.data_:" <<
    to_string(other.data_);
}

MoveOnlyLight& MoveOnlyLight::operator=(MoveOnlyLight&& other)
{
  data_ = other.data_;
  other.data_ = -2;

  ++move_assign_counter_; 

  clog <<
    move_assign_message <<
    ":" <<
    to_string(move_assign_counter_) <<
    "data_:" <<
    to_string(data_) <<
    "other.data_:" <<
    to_string(other.data_);

  return *this;
}

MoveOnlyLight::~MoveOnlyLight()
{
  if (data_ < 1)
  {
    clog << "MoveOnlyLight destructs for data_:" << to_string(data_);
  }

  ++dtor_counter_;

  clog << dtor_message << ":" << to_string(dtor_counter_) << "data_:" <<
    to_string(data_);
}

MoveOnlyLight return_rvalue_move_only_light(const int value)
{
  MoveOnlyLight a {value};

  return std::move(a);
}

const string CustomDestructorLight::move_ctor_message {
  "CustomDestructorLight move constructs"};
const string CustomDestructorLight::dtor_message {
  "CustomDestructorLight destructs"};

// cf. https://stackoverflow.com/questions/16284629/undefined-reference-to-static-variable-c
// Need to provide definition of that data member.
int CustomDestructorLight::move_ctor_counter_ {0};
int CustomDestructorLight::dtor_counter_ {0};

CustomDestructorLight::CustomDestructorLight(const int value):
  data_{value}
{
  if (value < 1)
  {
    throw std::runtime_error(
      "Runtime Error; CustomDestructorLight ctor: input value: " +
      to_string(value) +
      "less than 1 and thus invalid.");
  }
}

CustomDestructorLight::CustomDestructorLight(CustomDestructorLight&& other):
  data_{other.data_}
{
  other.data_ = -1;

  ++move_ctor_counter_; 

  clog <<
    move_ctor_message <<
    ":" <<
    to_string(move_ctor_counter_) <<
    "data_:" <<
    to_string(data_) <<
    "other.data_:" <<
    to_string(other.data_);
}

CustomDestructorLight::~CustomDestructorLight()
{
  if (data_ < 1)
  {
    clog << "CustomDestructorLight destructs for data_:" << to_string(data_);
  }

  ++dtor_counter_;

  clog << dtor_message << ":" << to_string(dtor_counter_) << "data_:" <<
    to_string(data_);
}


const string CustomDestructorEncapsulated::dtor_message {
  "CustomDestructorEncapsulated destructs"};

int CustomDestructorEncapsulated::dtor_counter_ {0};

CustomDestructorEncapsulated::CustomDestructorEncapsulated(
  const int value1,
  const int value2
  ):
  data_{value1},
  other_data_{value2}
{
  if (value1 < 1 || value2 < 1)
  {
    throw std::runtime_error(
      "Runtime Error; CustomDestructorEncapsulated ctor: input value: " +
      to_string(value1) +
      " or " +
      to_string(value2) +
      "less than 1 and thus invalid.");
  }
}

CustomDestructorEncapsulated::~CustomDestructorEncapsulated()
{
  if (data_.data() < 1 || other_data_ < 1)
  {
    clog << "CustomDestructorEncapsulated destructs for data_:" <<
      to_string(data_.data()) << " or " << to_string(other_data_);
  }

  ++dtor_counter_;

  clog << dtor_message << ":" << to_string(dtor_counter_) << "data_:" <<
    to_string(data_.data()) << " or " << to_string(other_data_);
}


} // namespace Classes
} // namespace Cpp
