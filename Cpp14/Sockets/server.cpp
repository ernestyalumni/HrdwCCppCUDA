// \name server.cpp
// \ref https://www.geeksforgeeks.org/socket-programming-cc/
// COMPILATION TIP 
// g++ -std=c++14 server.cpp -o server


// \ref http://pubs.opengroup.org/onlinepubs/7908799/xsh/unistd.h.html
#include <unistd.h> // ::close, ::sleep, ::unlink
#include <iostream>
#include <sys/socket.h>
#include <stdlib.h> // size_t, NULL, EXIT_FAILURE, EXIT_SUCCESS, ::exit
// \ref http://pubs.opengroup.org/onlinepubs/7908799/xns/netinetin.h.html
#include <netinet/in.h> // sockaddr_in struct 
#include <string>

#include <array>

#include <stdexcept>

constexpr const uint PORT {8080};

bool booleanCheckFd(const int fd)
{
  if (fd == 0)
  {
    return false;
  }
  else
  {
  return true;
  }
}

void checkFdThrow(const int fd)
{
  if (fd == 0)
  {
    // \ref https://stackoverflow.com/questions/12171377/how-to-convert-errno-to-exception-using-system-error
    // Take a look, interesting that it'll take error codes such as EFAULT, EDOM
    // EBADF - bad file number
    // http://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html
    throw std::system_error(EBADF, std::generic_category(), "socket failed");
  }
}

void checkFdExit(const int fd)
{
  if (fd == 0)
  {
//    ::perror("socket failed"); // print system error // stdio.h
    std::cerr << "socket failed";
    ::exit(EXIT_FAILURE);
  }
}


void setSocketOptions(const int fd, const int option)
{
  auto set_socket_option_result {
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, 
      &option, sizeof(option))
  };
  std::cout << "\n set_socket_option_result : " << set_socket_option_result <<
    '\n';

  if (set_socket_option_result)
  {
    throw std::system_error(EBADF, std::generic_category(), "socket failed");
  }
}



int main()
{
  int server_fd, new_socket, valread;
  struct sockaddr_in address; // from netinet/in.h
  int opt = 1;
  int addrlen = sizeof(address);

  std::cout << "\n addrlen : " << addrlen << '\n';

  constexpr const uint buffer_size {1024};
  std::array<char, buffer_size> buffer;
  buffer.fill(0);

  for (auto iter{buffer.begin()}; iter < (buffer.begin() + 10); iter++)
  {
    std::cout << *iter << ' ';
  }

  std::string hello {"Hello from server"};

  // Creating socket file descriptor
  server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  std::cout << "\n server_fd : " << server_fd << '\n';

  checkFdThrow(server_fd);

  // set socket options

  setSocketOptions(server_fd, opt); // 0 is the expected result

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = ::htons(PORT);

  // Forcefully attaching socket to the port 8080
//https://linux.die.net/man/3/htons
}

