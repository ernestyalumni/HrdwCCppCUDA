# Sockets; socket programming 

cf. [Socket Programming in C/C++, GeeksforGeeks](https://www.geeksforgeeks.org/socket-programming-cc/)

### What is socket programming? 

Socket programming is a way of connecting 2 nodes on a network to communicate with each other. 
- 1 socket (node) listens on a specific port at an IP (server listens), while 
- another socket reaches out to the other to form a connection.
- server forms the listener socket while 
- client reaches out to server

#### Stages for server 

cf. [`socket`(2): create endpoint for communication - Linux man page](https://linux.die.net/man/2/socket)

* **Socket creation**
``` 
#include <sys/types.h> //
#include <sys.socket.h>

int socket(int domain, int type, int protocol);
``` 
  - `socket()` creates an endpoint for communication and returns a descriptor 
    * *domain* argument specifies a communication domain; this selects protocol family which will be used for communication. These families are defined in `<sys.socket.h>`, including  
      
      | Name | Purpose | Man page |
      | :--- | :------ | :------- |
      | `AF_UNIX`, `AF_LOCAL` | Local communication | [unix(7)](https://linux.die.net/man/7/unix) | 
      | `AF_INET` | IPv4 Internet protocols | [ip(7)](https://linux.die.net/man/7/ip) | 
      | `AF_INET6` | IPv6 Internet protocols | [ipv6(7)](https://linux.die.net/man/7/ipv6) | 

The socket has the indicated *type*, which specifies the communication semantics. Currently defined types are: 

`SOCK_STREAM` - provides sequenced, reliable, 2-way, connection-based byte stream. An out-of-band data transmission mechanism may be supported.  
`SOCK_DGRAM` - supports datagrams (connectionless, unreliable messages of a fixed maximum length). 
`SOCK_SEQPACKET` - provides a sequenced, reliable, 2-way connection-based data transmission path for datagrams of fixed maximum length; a consumer is required to read an entire packet with each input system call. 
`SOCK_RAW` - provides raw network protocol access. 

Since Linux 2.6.27, *type* argument, in addition to specifying a socket type, it may include bitwise OR of any of the following values, to modify the behavior of `socket()`: 

**SOCK_NONBLOCK** - set the `O_NONBLOCK` file status flag on new open file description. Using this flag saves extra calls to `fcntl(2)` to achieve the same result. 

**SOCK_CLOEXEC** - set the close-on-exec (`FD_CLOEXEC`) flag on new file descriptor.

*protocol* specifies protocol to be used with socket. Normally, only a single protocol exists to support a particular socket type within a given protocol family, in which case *protocol* can be specified as 0. 

cf. https://linux.die.net/man/2/socket

* **`::setsockopt`** 

  - this helps in manipulating options for socket referred by the file descriptor `sockfd`. This is completely optional, but it helps in reuse of address and port. Prevents error such as: "address already in use."

``` 
#include <sys/socket.h>

int setsockopt(int socket, int level, int option_name,
  const void *option_value, socklen_t option_len);
``` 
  - The `setsockopt()` function shall set option specified by the `option_name` argument, at the protocol level specified by the `level` argument, to the value pointed to by the `option_value` argument for the socket associated with the fd specified by the `socket` argument. 
  - *Return Value*: upon successful completion, `setsockopt()` shall return 0. Otherwise, -1 shall be returned and *errno* set to indicate the error  

cf. http://pubs.opengroup.org/onlinepubs/009695399/functions/setsockopt.html


* **`::bind`**  bind a name to a socket
``` 
#include <sys/types.h>  
#include <sys/socket.h>

int bind(int sockfd, const struct sockaddr *addr,
  socklen_t addrlen);
``` 
  - When a socket is created with `socket`(2), it exists in a name space (address family) but has no address assigned to it. `bind`() assigns the address specified by `addr` to the socket referred to by the fd `sockfd`. `addrlen` specifies size, in bytes, of the address structure pointed to by `addr`. 
  - *Return Value* - on success, 0 is returned. On error, -1 is returned, and *errno* is set appropriately. 

  cf. https://linux.die.net/man/2/bind 

  - after creation of the socket, bind function binds the socket to the address and port number specified in `addr` (custom data structure). In the example code, we bind the server to the localhost, hence we use `INADDR_ANY` to specify the IP address. 

* **`::listen`**  listen for connections on a socket 
``` 
#include <sys/types.h> 
#include <sys/socket.h>

int listen(int sockfd, int backlog);
```  

  - `listen()` marks the socket referred to by `sockfd` as a passive socket, that is, as a socket that'll be used to accept incoming connection requests using `accept(2)`.  
  - `sockfd` argument is a fd that refers to a socket of type `SOCK_STREAM` or `SOCK_SEQPACKET` 
  - `backlog` argument defines the max. length to which the queue of pending connections for `sockfd` may grow. If a connection request arrives when the queue is full, client may receive an error with an indication of `ECONNREFUSED` or, if underlying protocol supports retransmission, the request may be ignored so that a later reattempt at connection succeeds.
  - *Return Value* - on success, 0 is returned. On error, -1 is returned, and *errno* set appropriately.

* **`::accept`** - accept a connection on a socket 

``` 
#include <sys/types.h>
#include <sys/socket.h>

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
```   
  - `accept()` system call used with connection-based socket types (`SOCK_STREAM`, `SOCK_SEQPACKET`). 
    * it extracts the 1st connection request on the queue of pending connections for the listening socket, `sockfd`, creates a new connected socket, and returns a new fd referring to that socket. 
      - Newly created socket is not in listening state. 
      - original socket `sockfd` unaffected 
    * argument `sockfd` is a socket that has been created with `socket`, bound to a local address with `bind` and is listening for connections after a `listen`. 
    - `addr` is a pointer to `sockaddr` structure, filled with address of peer socket, as known to communications layer; exact format of address returned `addr` is determined by the socket's address family. 
    - `addrlen` is a value-result argument; caller must initialize it to contain the size (in bytes) of the structure pointed to by `addr`; on return it'll contain the actual size of the peer address. 
  - *Return Value* - on success, these system calls return a nonnegative integer that's a descriptor for the accepted socket. On error, -1 is returned and *errno* is set appropriately. 
  - extracts 1st connection request on queue of pending connections for the listening socket, `sockfd`, creates a new, connected socket, and returns a new fd referring to that socket. AT this point, connection is established between client and server, and they're ready to transfer data. 

#### Stages for Client

* **Socket connection** - exactly same as that of server's socket creation 
* **`::connect`** - initiate a connection on a socket 
``` 
#include <sys/types.h>
#include <sys/socket.h>

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
``` 
  - `connect()` system call connects socket referred to by the fd `sockfd` to address specified by `addr`
    * `addrlen` argument specifies size of `addr` 
      - format of the address in `addr` determined by the address space of the socket `sockfd` 
    * if socket `sockfd` is of type `SOCK_DGRAM` then `addr` is address to which datagrams are sent by default, and only address from which datagrams are received. 
    - if socket is of type `SOCK_STREAM` or `SOCK_SEQPACKET`, this call attempts to make a connection to the socket that's bound to the address specified by `addr` 
  cf. https://linux.die.net/man/2/connect 





## Interesting links

https://en.wikipedia.org/wiki/Network_socket 

https://www.binarytides.com/socket-programming-c-linux-tutorial/

https://linux.die.net/man/3/htons

https://www.geeksforgeeks.org/socket-programming-cc/

### Google search key words 

socket linux C programming






