//------------------------------------------------------------------------------
/// \file InternetAddress_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "IPC/Sockets/InternetAddress.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "Utilities/ToBytes.h"

#include <boost/test/unit_test.hpp>
#include <netinet/ip.h>
#include <string>

using Cpp::Utilities::TypeSupport::get_underlying_value;
using IPC::Sockets::Domain;
using IPC::Sockets::InternetAddress;
using IPC::Sockets::InternetSocketAddress;
using IPC::Sockets::address_to_network_binary;
using Utilities::ToBytes;

BOOST_AUTO_TEST_SUITE(IPC)
BOOST_AUTO_TEST_SUITE(Sockets)
BOOST_AUTO_TEST_SUITE(InternetAddress_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(InternetSocketAddressConstructsWithHostByteOrdering)
{
	{
		InternetSocketAddress address;
		
		BOOST_TEST(address.sin_family == get_underlying_value(Domain::ipv4));

		const ToBytes to_bytes_sin_family {address.sin_family};
		const ToBytes to_bytes_sin_port {address.sin_port};
		const ToBytes to_bytes_sin_addr {address.sin_addr};

		to_bytes_sin_family.increasing_addresses_print();
		BOOST_TEST((to_bytes_sin_family.increasing_addresses_hex() == "20"));
	}
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AddressToNetworkBinaryConvertsAddressIntoBinaryForm)
{
	{
		std::string host_address {"127.0.0.1"};
		InternetSocketAddress address {21234};
		auto binary_result = 
			address_to_network_binary(host_address, address);
		BOOST_TEST_REQUIRE(static_cast<bool>(binary_result));
	}
}

BOOST_AUTO_TEST_SUITE_END() // InternetAddress_tests
BOOST_AUTO_TEST_SUITE_END() // Sockets
BOOST_AUTO_TEST_SUITE_END() // IPC