//------------------------------------------------------------------------------
/// \file CaptureCout.cpp
/// \brief Help capture std::cout standard output.
//------------------------------------------------------------------------------
#include "CaptureCout.h"

#include <functional>
#include <iostream>
#include <sstream> // std::ostringstream
#include <streambuf>
#include <utility> // std::pair

using std::cout;
using std::make_pair;
using std::ostringstream;
using std::pair;
using std::reference_wrapper;
using std::streambuf;

namespace Tools
{

//------------------------------------------------------------------------------
/// \details
/// cf. https://en.cppreference.com/w/cpp/io/basic_ostringstream
/// ostringstream effectively stores an instance of std string and performs
/// output operations to it.
/// rdbuf() returns associated stream buffer.
/// rdbuf(streambuf* sb) sets associated stream buffer to sb. Returns associated
/// stream buffer before operation. If there's no associated stream buffer,
/// returns null pointer.
//------------------------------------------------------------------------------

pair<
  reference_wrapper<ostringstream>,
  streambuf*> capture_cout()
{
  ostringstream local_oss;

  streambuf* cout_buffer {cout.rdbuf()}; // Save pointer to std::cout buffer

  cout.rdbuf(local_oss.rdbuf());

  return make_pair<
    reference_wrapper<ostringstream>,
    streambuf*>(
      local_oss,
      move(cout_buffer));
}

streambuf* capture_cout(ostringstream& local_oss)
{
  streambuf* cout_buffer {cout.rdbuf()};

  cout.rdbuf(local_oss.rdbuf());

  return cout_buffer;
}

CaptureCout::CaptureCout():
  oss_{}//,
  //cout_buffer_ptr_{new streambuf}
{
  cout_buffer_ptr_ = cout.rdbuf();
}

CaptureCout::~CaptureCout()
{
  //delete cout_buffer_ptr_;
}

void CaptureCout::capture_locally()
{
  cout.rdbuf(oss_.rdbuf());
  //return oss_;
}

/*
void CaptureCout::operator()()
{
  capture_locally();
}*/

void CaptureCout::restore_cout()
{
  cout.rdbuf(cout_buffer_ptr_);
}

} // namespace Tools

