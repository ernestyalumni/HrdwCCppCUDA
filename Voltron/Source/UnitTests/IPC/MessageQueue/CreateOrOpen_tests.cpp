//------------------------------------------------------------------------------
/// \file CreateOrOpen_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/MessageQueue/CreateOrOpen.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "IPC/MessageQueue/DataStructures.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IPC::MessageQueue::CreateOrOpen;
using IPC::MessageQueue::AdditionalOperationFlags;
using IPC::MessageQueue::ModePermissions;
using IPC::MessageQueue::OperationFlags;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(MessageQueue)
BOOST_AUTO_TEST_SUITE(CreateOrOpen_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CreateOrOpenCreatesNewMessageQueue)
{
  {
    // Name of the POSIX object referencing the queue.
    const std::string message_queue_object_name {"/myqueue123"};

    // Max length of a message (just for this process)
    constexpr long maximum_message_length {70};

    const int operation_flag {
      get_underlying_value(OperationFlags::send_and_receive) ||
        get_underlying_value(AdditionalOperationFlags::create) ||
        get_underlying_value(AdditionalOperationFlags::exclusive_existence)};

    CreateOrOpen create_q {
      message_queue_object_name,
      operation_flag,
      get_underlying_value(ModePermissions::user_rwx)};

    auto result = create_q(true);
    BOOST_TEST(!(static_cast<bool>(result.first)));
    BOOST_TEST((*(result.second)) > -1);

    std::cout <<
      "\n\n CreateOrOpenCreatesNewMessageQueue: message queue descriptor: " <<
        *(result.second) << "\n";
  }
}

BOOST_AUTO_TEST_SUITE_END() // CreateOrOpen_tests
BOOST_AUTO_TEST_SUITE_END() // MessageQueue
BOOST_AUTO_TEST_SUITE_END() // IPC