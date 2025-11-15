//------------------------------------------------------------------------------
/// \file bind_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  The following example shows how to bind a stream socket in UNIX
///   (AF_UNIX) domain, and accept connections.
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html     
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -I ../../ -std=c++14 Socket.cpp Bind_main.cpp ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o Bind_main
//------------------------------------------------------------------------------
#include <sys/socket.h>
#include <sys/un.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MY_SOCK_PATH "/somepath"
#define LISTEN_BACKLOG 50

#define handle_error(msg) \
   do { perror(msg); exit(EXIT_FAILURE); } while (0)

int main(int argc, char *argv[])
{
  int sfd, cfd;
  struct sockaddr_un my_addr, peer_addr;
  socklen_t peer_addr_size;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sfd == -1)
  {
    handle_error("socket");
  }

  memset(&my_addr, 0, sizeof(struct sockaddr_un));
                   /* Clear structure */
  my_addr.sun_family = AF_UNIX;
  strncpy(my_addr.sun_path, MY_SOCK_PATH,
       sizeof(my_addr.sun_path) - 1);


  if (bind(sfd, (struct sockaddr *) &my_addr,
    sizeof(struct sockaddr_un)) == -1)
  {
    handle_error("bind");
  }

  if (listen(sfd, LISTEN_BACKLOG) == -1)
   handle_error("listen");

  /* Now we can accept incoming connections one
  at a time using accept(2) */

  peer_addr_size = sizeof(struct sockaddr_un);
  cfd = accept(sfd, (struct sockaddr *) &peer_addr,
            &peer_addr_size);
  if (cfd == -1)
   handle_error("accept");

  /* Code to deal with incoming connection(s)... */

  /* When no longer required, the socket pathname, MY_SOCK_PATH
  should be deleted using unlink(2) or remove(3) */
}
