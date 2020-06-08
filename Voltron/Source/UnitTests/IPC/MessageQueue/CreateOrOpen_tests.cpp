//------------------------------------------------------------------------------
/// \file CreateOrOpen_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/MessageQueue/CreateOrOpen.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "IPC/MessageQueue/DataStructures.h"
#include "IPC/MessageQueue/MessageQueueDescription.h"
#include "Tools/TemporaryDirectory.h"

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <string>

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IPC::MessageQueue::AdditionalOperationFlags;
using IPC::MessageQueue::Close;
//using IPC::MessageQueue::CreateOrOpen;
using IPC::MessageQueue::ModePermissions;
using IPC::MessageQueue::NoAttributesOpen;
using IPC::MessageQueue::OpenConfiguration;
using IPC::MessageQueue::OperationFlags;
using IPC::MessageQueue::Unlink;
using Tools::TemporaryDirectory;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(MessageQueue)

BOOST_AUTO_TEST_SUITE(OpenConfiguration_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(OpenConfigurationConstructsWithNameAndIntegerOperationFlag)
{
  const OpenConfiguration config {
    "/myqueue123",
    get_underlying_value(OperationFlags::send_and_receive)};

  BOOST_TEST(config.name() == "/myqueue123");
  BOOST_TEST(config.operation_flag() ==
    get_underlying_value(OperationFlags::send_and_receive));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithNameAndOperationFlag)
{
  TemporaryDirectory temp_dir {"temp"};
  const OpenConfiguration config {temp_dir.path(), OperationFlags::send_only};

  BOOST_TEST(config.name() == temp_dir.path());
  BOOST_TEST(config.operation_flag() ==
    get_underlying_value(OperationFlags::send_only));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddAdditionalOperationOrsOperationFlag)
{
  TemporaryDirectory temp_dir {"temp"};
  OpenConfiguration config {temp_dir.path(), OperationFlags::receive_only};

  BOOST_TEST_REQUIRE(config.name() == temp_dir.path());
  BOOST_TEST_REQUIRE(config.operation_flag() ==
    get_underlying_value(OperationFlags::receive_only));

  config.add_additional_operation(AdditionalOperationFlags::nonblocking);

  BOOST_TEST(config.operation_flag() !=
    get_underlying_value(OperationFlags::receive_only));

  BOOST_TEST((config.operation_flag() |
    get_underlying_value(OperationFlags::receive_only)));

  BOOST_TEST((config.operation_flag() |
    get_underlying_value(AdditionalOperationFlags::nonblocking)));

  BOOST_TEST(!(config.operation_flag() &
    get_underlying_value(AdditionalOperationFlags::create)));

  BOOST_TEST((config.operation_flag() |
    get_underlying_value(AdditionalOperationFlags::create)));

  BOOST_TEST((config.operation_flag() |
    get_underlying_value(AdditionalOperationFlags::exclusive_existence)));

  // cf. https://stackoverflow.com/questions/18591924/how-to-use-bitmask
  // http://www.dylanleigh.net/notes/c-cpp-tips/
  // Test for a flag. AND with the bitmask before testing with ==.

  BOOST_TEST((config.operation_flag() & 
    get_underlying_value(OperationFlags::receive_only)) ==
      get_underlying_value(OperationFlags::receive_only));

  BOOST_TEST((config.operation_flag() & 
    get_underlying_value(AdditionalOperationFlags::nonblocking)) ==
      get_underlying_value(AdditionalOperationFlags::nonblocking));

  BOOST_TEST((config.operation_flag() & 
    get_underlying_value(AdditionalOperationFlags::create)) !=
      get_underlying_value(AdditionalOperationFlags::create));

  // Test for multiple flags. OR the bitmasks.
  BOOST_TEST((config.operation_flag() &
    (get_underlying_value(OperationFlags::receive_only) |
      get_underlying_value(AdditionalOperationFlags::nonblocking))) ==
        (get_underlying_value(OperationFlags::receive_only) |
          get_underlying_value(AdditionalOperationFlags::nonblocking)));
}

BOOST_AUTO_TEST_SUITE_END() // OpenConfiguration_tests

BOOST_AUTO_TEST_SUITE(NoAttributesOpen_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithOpenConfiguration)
{
  TemporaryDirectory temp_dir {"temp"};
  OpenConfiguration config {temp_dir.path(), OperationFlags::send_and_receive};

  NoAttributesOpen open_object {config};

  BOOST_TEST(open_object.configuration().name() == temp_dir.path());
  BOOST_TEST(open_object.configuration().operation_flag() ==
    get_underlying_value(OperationFlags::send_and_receive));
  BOOST_TEST(!static_cast<bool>(open_object.message_queue_descriptor()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CallOperatorFailsToOpenQueueWithoutCreateFlag)
{
  TemporaryDirectory temp_dir {"temp"};
  OpenConfiguration config {temp_dir.path(), OperationFlags::send_and_receive};

  NoAttributesOpen open_object {config};

  BOOST_TEST_REQUIRE(open_object.configuration().name() == temp_dir.path());
  BOOST_TEST_REQUIRE(open_object.configuration().operation_flag() ==
    get_underlying_value(OperationFlags::send_and_receive));
  BOOST_TEST_REQUIRE(
    !static_cast<bool>(open_object.message_queue_descriptor()));

  auto result = open_object();

  BOOST_TEST(static_cast<bool>(result.first));
  BOOST_TEST(!static_cast<bool>(result.second));
  BOOST_TEST(!static_cast<bool>(open_object.message_queue_descriptor()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(CallOperatorFailsToOpenQueueWithCreateFlag)
{
  TemporaryDirectory temp_dir {"temp"};
  OpenConfiguration config {temp_dir.path(), OperationFlags::send_and_receive};
  config.add_additional_operation(AdditionalOperationFlags::create);

  NoAttributesOpen open_object {config};

  BOOST_TEST_REQUIRE(open_object.configuration().name() == temp_dir.path());
  BOOST_TEST_REQUIRE(
    !static_cast<bool>(open_object.message_queue_descriptor()));

  auto result = open_object();

  BOOST_TEST(static_cast<bool>(result.first));
  BOOST_TEST(!static_cast<bool>(result.second));
  BOOST_TEST(!static_cast<bool>(open_object.message_queue_descriptor()));

  //auto close_result = Close()(*(open_object.message_queue_descriptor()));
  //BOOST_TEST(!static_cast<bool>(close_result));
  //auto unlink_result = Unlink()(open_object.configuration().name());
  //BOOST_TEST(!static_cast<bool>(unlink_result));
}


BOOST_AUTO_TEST_SUITE_END() // NoAttributesOpen_tests

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

    /*
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
    */
  }
}

BOOST_AUTO_TEST_SUITE_END() // CreateOrOpen_tests
BOOST_AUTO_TEST_SUITE_END() // MessageQueue
BOOST_AUTO_TEST_SUITE_END() // IPC