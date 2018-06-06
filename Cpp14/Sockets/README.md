# Sockets; socket programming 

cf. [Socket Programming in C/C++, GeeksforGeeks](https://www.geeksforgeeks.org/socket-programming-cc/)

### What is socket programming? 

Socket programming is a way of connecting 2 nodes on a network to communicate with each other. 
- 1 socket (node) listens on a specific port at an IP (server listens), while 
- another socket reaches out to the other to form a connection.
- server forms the listener socket while 
- client reaches out to server

## Client-Server communication 

cf. [Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)

### Server 

- passively waits for and responds to clients (listens)
- passive socket 

### Client 

- initiates communication
- must know address and port of the server
- active socket 

### Sockets - Procedures 

| **Primitive** | **Meaning** |
| :------------ | :---------- |
| Socket  | Create a new communication endpoint |
| Bind    | Attach a local address to a socket  |
| Listen  | Announce willingness to accept connections |
| Accept  | Block caller until a connection request arrives |
| Connect | Actively attempt to establish a connection |
| Send    | Send some data over the connection |
| Receive | Receive some data over the connection |
| Close  | Release the connection |


Stream (e.g. TCP) = { Server, Client }

Server = (`socket()`, `bind()`, `listen()`, `accept()`, { (`recv()`, `send()`) }, `close()`)

Client = (`socket()`, `connect()`, { (`send()`,`recv()` ) }, `close()`)

s.t.  

- *synchronization point* - (Client, `connect()`) : (Server,`accept()`) |-> synchronization point.

- (Client, `send()`) : (Server, `recv()`)
- (Server, `send()`) : (Client, `recv()`)

and cycles

- (Server, `recv()`) : (Server, `send()`)
- (Client, `send()`) : (Client, `recv()`)

Datagram (e.g. UDP) = { Server, Client }

Server = (`socket()`, `bind()`, { (`recvfrom()`, `sendto()`) }, `close()`)

Client = (`socket()`, `bind()`, { (`sendto()`,`recvfrom()` ) }, `close()`)

s.t.  

- (Client, `sendto()`) : (Server, `recvfrom()`)
- (Server, `sendto()`) : (Client, `recvfrom()`)

and cycles

- (Server, `recvfrom()`) : (Server, `sendto()`)
- (Client, `sendto()`) : (Client, `recvfrom()`)


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

The socket has the indicated *type*, which specifies the communication semantics. It's a communication type. Currently defined types are: 

`SOCK_STREAM` - provides sequenced, reliable, 2-way, connection-based byte stream. An out-of-band data transmission mechanism may be supported.  
`SOCK_DGRAM` - supports datagrams (connectionless, unreliable messages of a fixed maximum length). 
`SOCK_SEQPACKET` - provides a sequenced, reliable, 2-way connection-based data transmission path for datagrams of fixed maximum length; a consumer is required to read an entire packet with each input system call. 
`SOCK_RAW` - provides raw network protocol access. 

Since Linux 2.6.27, *type* argument, in addition to specifying a socket type, it may include bitwise OR of any of the following values, to modify the behavior of `socket()`: 

**SOCK_NONBLOCK** - set the `O_NONBLOCK` file status flag on new open file description. Using this flag saves extra calls to `fcntl(2)` to achieve the same result. 

**SOCK_CLOEXEC** - set the close-on-exec (`FD_CLOEXEC`) flag on new file descriptor.

*protocol* specifies protocol to be used with socket. Normally, only a single protocol exists to support a particular socket type within a given protocol family, in which case *protocol* can be specified as 0. 
  - `IPPROTO_TCP`, `IPPROTO_UDP` 

cf. https://linux.die.net/man/2/socket

```
int sockid = ::socket(int domain, int type, int protocol)
```
- `sockid` - socket *descriptor*, an integer (like a file-handle)
- `domain` or "family": integer, communication domain, e.g. 
  * `PF_INET`, IPv4 protocols, Internet addresses (typically used)
  * `PF_UNIX`, Local communication, File addresses
- `type` - communicate type 
- upon failure, returns -1

NOTE: socket call doesn't specify where data will be coming from, nor where it will be going to - it just creates the interface! 

cf. [Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)


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


* **`::bind`**  bind a name to a socket; associates and reserves a port for use by socket
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

  - `addr` the (IP) address and port of machine 
    * for TCP/IP server, internet address usually set to `INADDR_ANY`, i.e. chooses any incoming interface.

cf. [Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)

- bind can be skipped for both types of sockets
- Datagram socket:
  * if only sending, no need to bind. OS finds a port each time socket sends a packet.
  * if receiving, need to bind.
- Stream socket:
  * destination determined during connection setup
  * don't need to know port sending from (during connection setup, receiving end is informed of port) 



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


- Instructs TCP protocol implementation to listen for connections

- `listen()` is **non-blocking**: returns immediately

- The listening socket (`sockid`)
  * is never used for sending and receiving 
  * is used by the server only as a way to get new sockets.

cf. [Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)


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

`sockfd` - integer, the new socket (used for data-transfer)
`accept()` is blocking: waits for connection before returning;  
dequeues next connection on queue for socket (sockid) 


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

cf. [Socket Programming in C/C++, GeeksforGeeks](https://www.geeksforgeeks.org/socket-programming-cc/)

client establishes a connection with server by calling `connect()` 

`addr` - address of passive participant
`connect()` is blocking 

cf. [Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)


## `::close()` - close - close a file descriptor

``` 
#include <unistd.h>
int close(int fd)
```
`close()` closes a file descriptor so that it no longer refers to any file and may be reused. 
  - any record locks (see `fcntl`) held on file it was associated with, and owned by the process, are removed (regardless of the fd that was used to obtain lock)
  - If `fd` is the last file descriptor referring to the underlying open fd, the resources associated with the open file description are freed;
  - if descriptor was the last reference to a file which has been removed using `unlink`, file is deleted.

Note, not checking the return value of `close()` is a common, but nevertheless serious programming error; it's quite possible that errors on a previous `write` operation are 1st. reported at the final `close()`.  Not checking the return value when closing the file may lead to silent loss of data. 

#### Return Value (of `::close`)

`close()` returns zero on success; on error, -1 is returned and `errno` is set appropriately. 

#### Errors (of `::close`)

`EBADF` - `fd` isn't a valid open file descriptor 
`EINTR` - `close()` call was interrupted by a signal 
`EIO` - an I/O error occurred.

### Socket close in C: `::close()` 

- when finished using a socket, socket should be closed
- closing a socket 
  * closes a connection (for stream socket)
  * frees up port used by socket.

cf. https://linux.die.net/man/2/close

## Specifying Addresses, `sockaddr`, vs. `sockaddr_in`   

* Socket API defines a *generic* data type for addresses: 

From `ip(7)`, 

#### Address format 

An IP socket address is defined as a combination of an IP interface address, and 16-bit port number. 
The basic IP protocol doesn't supply port numbers; they're implemented by higher level protocols like `udp(7)` and `tcp(7)`. 
On raw sockets `sin_port` is set to the IP protocol. 

Actual structure passed for `addr` argument in `bind`

``` 
struct sockaddr
{
  sa family_t sa_family;
  char        sa_data[14];
}
``` 
The only purpose of this structure is to cast the structure pointer passed in `addr` in order to avoid compiler warnings. 

cf. http://man7.org/linux/man-pages/man2/bind.2.html

``` 
struct sockaddr
{
  unsigned short sa_family; // Address family (e.g. AF_INET)
  char sa_data[14];         // Family-specific address information
}
```

Particular form of `sockaddr` used for TCP/IP addresses:

``` 
struct in_addr
{
  unsigned long s_addr; // Internet address (32-bits)
} 

struct sockaddr_in 
{
  unsigned short sin_family   // Internet protocol (AF_INET)
  unsigned short sin_port;    // Address port (16 bits)
  struct in_addr sin_addr;    // Internet address (32 bits)
  char sin_zero[8];           // Not used
}
``` 

**Important** `sockaddr_in` can be casted to a `sockaddr`

For `ip` 

``` 
struct sockaddr_in 
{
  sa_family_t sin_family; // address family: AF_INET 
  in_port_t   sin_port;   // port in network byte order 
  struct in_addr sin_addr; // internet address
};


// Internet address
struct in_addr
{
  uint32_t s_addr;  // address in network byte order
};

``` 

* The `sin_addr` member is defined as the structure `in_addr`, which holds the IP number in network byte order. If you examine the structure `in_addr`, you'll see that it consists of 1 32-bit unsigned integer. 
* Finally, remainder of structure is padded to 16 bytes by the member `sin_zero[8]` for 8 bytes. This member doesn't require any initialization and isn't used. 

cf. http://www.ccplusplus.com/2011/10/struct-sockaddrin.html

cf. http://man7.org/linux/man-pages/man7/ip.7.html


Structure `sockaddr` is a generic contained that just allows OS to be able to read the 1st couple of bytes that identify the address family. Address family determines what variant of the `sockaddr` struct to use that contains elements that make sense for that specific communication type. 
  - For IP networking, we use `struct sockaddr_in`, which is defined in header `netinet/in.h`. 

  ```
  struct sockaddr_in 
  {
    __uint8_t     sin_len;
    sa_family_t   sin_family;
    in_port_t     sin_port;
    struct in_addr sin_addr;
    char          sin_zero[8];
  }
  ```

`sin_family` - address family we used when we set up the socket.
`sin_family` is always set to `AF_INET`. This is required; in Linux 2.2 most networking functions return `EINVAL` when this setting is missing. 
`sin_port` contains port in network byte order. 
  * port numbers below 1024 are called *privileged ports* (or sometimes: *reserved ports*). 
  * Only a privileged process (on Linux: a process that has the `CAP_NET_BIND_SERVICE` capability in user namespace governing its network namespace) may `bind` to these sockets.
  * note that raw IPv4 protocol as such has no concept of a port, they're implemented only by higher protocols like `tcp`, `udp`. 

`sin_addr` is IP host address. `s_addr` member of `struct in_addr` contains host interface address in network byte order. 
  - it's the address for this socket. e.g. This is just your machine's IP address. With IP, your machine will have 1 IP address for each network interface. 
    * e.g. if machine has both Wi-Fi and ethernet connections, that machine will have 2 addresses, 1 for each interface. 
    - Most of the time, we don't care to specify a specific interface and can let the operating system use whatever it wants. Special address for this is 0.0.0.0, defined by symbolic constant `INADDR_ANY`. 

##### **`INADDR_ANY`** 

`INADDR_ANY` is special IP address 0.0.0.0 which binds the transport endpoint to all IP addresses on the machine.  

cf. `demo-tcp-03`

cf. https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html 


`htonl, htons` - convert values between host and network byte order 

``` 
#include <arpa/inet.h>
``` 

cf. https://linux.die.net/man/3/htons


## `sendto` - send a message on a socket 

cf. http://pubs.opengroup.org/onlinepubs/007904875/functions/sendto.html 

```
#include <sys/socket.h>
ssize_t sendto(int socket, const void *message, size_t length,
  int flags, const struct sockaddr *dest_addr,
  socklen_t dest_len);
```

`sendto()` send a message through a connection-mode or connectionless-mode socket. If socket is connection-mode, `dest_addr` shall be ignored. 

arguments:
`socket` - specifies socket fd
`message` - points to a buffer containing message to be sent.
`length` - specfies size of message in bytes
`flags` - specifies type of message transmission. Values of this argument formed by logically OR'ing 0 or more of the following flags: 
  `MSG_EOR` - terminates a record (if supported by the protocol) 
  `MSG_OOB` - sends out-of-band data on sockets that support out-of-band data. Significance and semantics of out-of-band data are protocol-specific. 

`dest_addr` - points to a `sockaddr` structure containing destination address. Length and format of address depend on the address family of socket  
`dest_addr` specifies address of the target. 

`dest_len` specifies length of `sockaddr` structure pointed to by `dest_addr` argument. 

Returns, upon successful completion, `sendto()` shall return number of bytes sent. Otherwise, -1 shall be returned and `errno` set to indicate error.

## `recvfrom` - receive a message from a socket 



### `::getsockname` - get socket name 

cf. http://man7.org/linux/man-pages/man2/getsockname.2.html

``` 
#include <sys/socket.h>

int ::getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
```

`getsockname()` returns current address to which socket `sockfd` is bound, in the buffer pointed to by `addr`. The `addrlen` arugment should be initialized to indicate amount of space (in bytes) pointed to by `addr`. On return, it contains actual size of the socket address.

Returned address is truncated if buffer provided is too small. 

Returns - on success, 0 is returned, on error -1 is returned, and `errno` is set appropriately. 


## TCP vs. UDP

[Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)

- both use **port numbers** - application specific construct serving as a communication endpoint
  * port numbers, 16-bit unsigned integer, 0 to 65535
  * provides end-to-end transport
- UDP: User Datagram Protocol 
  * no acknowledgements
  * no retransmissions
  * out of order, duplicates possible
  * connectionless, i.e., app indicates destination for each packet 
  * UDP more commonly used for quick lookups, and single use query-reply actions 
- TCP: Transmission Control Protocol
  * reliable **byte-stream channel** (in order, all arrive, no duplicates)
    - similar to file I/O
  * flow control
  * connection-oriented
  * bidirectional
  * TCP used for services with large data capacity, and persistent connection

## Sockets 

[Prof. Panagiota Fatourou, CSD May 2012, "Introduction to Sockets Programming in C using TCP/IP"](http://www.csd.uoc.gr/~hy556/material/tutorials/cs556-3rd-tutorial.pdf)

- uniquely identified by 
  * internet address 
  * end-to-end protocol (e.g. TCP or UDP)
  * port number
- 2 types of (TCP/IP) sockets (?)
  * **Stream** (e.g. uses TCP)
    - provide reliable byte-stream service 
  * **Datagram** sockets (e.g. uses UDP)
    - provide best-effort datagram service
    - messages up to 65.500 bytes (??)
- Sockets extend the conventional UNIX I/O facilities
  * files descriptors for network communication
  * extended read and write system calls 

UDP sockets are bound to ports.



## Interesting links

https://en.wikipedia.org/wiki/Network_socket 

https://www.binarytides.com/socket-programming-c-linux-tutorial/

https://linux.die.net/man/3/htons

https://www.geeksforgeeks.org/socket-programming-cc/

https://courses.engr.illinois.edu/cs241/sp2012/

http://www.csd.uoc.gr/~hy556/material.html

https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html

https://www.binarytides.com/socket-programming-c-linux-tutorial/

### Google search key words 

socket linux C programming






