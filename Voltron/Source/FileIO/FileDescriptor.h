//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef FILE_IO_FILE_DESCRIPTOR_H
#define FILE_IO_FILE_DESCRIPTOR_H

namespace FileIO
{

class FileDescriptor
{
  public:  

    FileDescriptor() = delete;

    explicit FileDescriptor(const int fd);

    // No Copies, Moves.

    // Copy ctor.
    FileDescriptor(const FileDescriptor&) = delete;

    // Copy assignment.
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    // Move ctor.
    FileDescriptor(FileDescriptor&& other);

    // Move assignment.
    FileDescriptor& operator=(FileDescriptor&& other);

    virtual ~FileDescriptor();

    const int fd() const
    {
      return fd_;
    }

  private:

    int fd_;    
};

} // namespace FileIO

#endif // FILE_IO_FILE_DESCRIPTOR_H
