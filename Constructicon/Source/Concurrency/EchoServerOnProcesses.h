#ifndef CONCURRENCY_ECHO_SERVER_ON_PROCESSES_H
#define CONCURRENCY_ECHO_SERVER_ON_PROCESSES_H

namespace Concurrency
{

class EchoServerOnProcesses
{
  public:

    EchoServerOnProcesses();

  private:

    int listenfd_;
    int connfd_;
    int port_;
};

} // namespace Concurrency

#endif // CONCURRENCY_ECHO_SERVER_ON_PROCESSES_H