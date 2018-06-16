/**
 * @file   : FileOpen.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : opening, reading, writing a file as RAII 
 * @ref    : pp. 26 Ch. 2 File I/O; 
 *   Robert Love, Linux System Programming,  
 * @detail : Using RAII for files. 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
 * */
#ifndef _FILEOPEN_H_
#define _FILEOPEN_H_

//------------------------------------------------------------------------------
/// \ref http://pubs.opengroup.org/onlinepubs/009696699/basedefs/sys/types.h.html
/// \details pid_t, 
//------------------------------------------------------------------------------
#include <fcntl.h>  // O_APPEND, ...
#include <string>
#include <sys/types.h>

//------------------------------------------------------------------------------
/// \brief flags for access mode for ::open
/// \details flags argument is bitwise-OR; may be bitwise-ORed with 0 or more 
///   of the following values, modifying behavior of open request 
//------------------------------------------------------------------------------
enum class OpenAccessFlags
{
  append  = O_APPEND,
  async   = O_ASYNC, 
  close_on_exec = O_CLOEXEC,
  create  = O_CREAT,
  directIO = O_DIRECT, // file opened for direct I/O
  isdirectory = O_DIRECTORY, // if name isn't a directory, call to open() fails

  // when given with O_CREAT, flag will cause call to open() to fail if file
  // given by name already exists. This is used to prevent race conditions on 
  // file creation.
  // If O_CREAT is not also provided, this flag has no meaning. 
  exclusive_with_create = O_EXCL,
  large_file = O_LARGEFILE,

  nonblocking = O_NONBLOCK

  // if file exists, it's a regular file and if given flags allow for writing
  // file will be truncated to 0 length. Use of O_TRUNC on FIFO or terminal
  // device is ignored
  truncate_to_0 = O_TRUNC

};

//------------------------------------------------------------------------------
/// \brief flags for access mode for ::open
/// \details flags argument is bitwise-OR; may be bitwise-ORed with 0 or more 
///   of the following values, modifying behavior of open request 
//------------------------------------------------------------------------------
enum class OpenAccessFlags
{
  append  = O_APPEND,
  async   = O_ASYNC, 
  close_on_exec = O_CLOEXEC,
  create  = O_CREAT,
  directIO = O_DIRECT, // file opened for direct I/O
  isdirectory = O_DIRECTORY, // if name isn't a directory, call to open() fails

  // when given with O_CREAT, flag will cause call to open() to fail if file
  // given by name already exists. This is used to prevent race conditions on 
  // file creation.
  // If O_CREAT is not also provided, this flag has no meaning. 
  exclusive_with_create = O_EXCL,
  large_file = O_LARGEFILE,

  nonblocking = O_NONBLOCK

  // if file exists, it's a regular file and if given flags allow for writing
  // file will be truncated to 0 length. Use of O_TRUNC on FIFO or terminal
  // device is ignored
  truncate_to_0 = O_TRUNC

};



class FileOpen
{
  public:
    
    FileOpen() = delete;
    
    FileOpen(const std::string& filename);
  
  private:
    int fd_;
    std::string filename_;   
};


#endif // _FILEOPEN_H_
