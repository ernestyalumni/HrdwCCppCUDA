//------------------------------------------------------------------------------
/// \file MainDropOne.cpp
/// \author Ernest Yeung
/// \brief Main file to drope a message into a defined queue, creating it if
/// user requested. The message is associated a priority, possibly user defined.
///
/// \details int main() is needed.
/// \ref http://mij.oltrelinux.com/devel/unixprg/
///-----------------------------------------------------------------------------

#include <sys/types.h> // pid_t
#include <unistd.h> // ::get_pid

int main(int argc, char* argv[])
{

  // \ref https://www.man7.org/linux/man-pages/man2/getpid.2.html
  // \brief Returns the process ID (PID) of the calling process.
  const pid_t my_pid {::getpid()};
}


    mqd_t msgq_id;
    unsigned int msgprio = 0;
    pid_t my_pid = getpid();
    char msgcontent[MAX_MSG_LEN];
    int create_queue = 0;
    char ch;            /* for getopt() */
    time_t currtime;
    
    
    /* accepting "-q" for "create queue", requesting "-p prio" for message priority */
    while ((ch = getopt(argc, argv, "qi:")) != -1) {
        switch (ch) {
            case 'q':   /* create the queue */
                create_queue = 1;
                break;
            case 'p':   /* specify client id */
                msgprio = (unsigned int)strtol(optarg, (char **)NULL, 10);
                printf("I (%d) will use priority %d\n", my_pid, msgprio);
                break;
            default:
                printf("Usage: %s [-q] -p msg_prio\n", argv[0]);
                exit(1);
        }
    }
    
    /* forcing specification of "-i" argument */
    if (msgprio == 0) {
        printf("Usage: %s [-q] -p msg_prio\n", argv[0]);
        exit(1);
    }
    
    /* opening the queue        --  mq_open() */
    if (create_queue) {
        /* mq_open() for creating a new queue (using default attributes) */
        msgq_id = mq_open(MSGQOBJ_NAME, O_RDWR | O_CREAT | O_EXCL, S_IRWXU | S_IRWXG, NULL);
    } else {
        /* mq_open() for opening an existing queue */
        msgq_id = mq_open(MSGQOBJ_NAME, O_RDWR);
    }
    if (msgq_id == (mqd_t)-1) {
        perror("In mq_open()");
        exit(1);
    }

    /* producing the message */
    currtime = time(NULL);
    snprintf(msgcontent, MAX_MSG_LEN, "Hello from process %u (at %s).", my_pid, ctime(&currtime));
    
    /* sending the message      --  mq_send() */
    mq_send(msgq_id, msgcontent, strlen(msgcontent)+1, msgprio);
    
    /* closing the queue        -- mq_close() */
    mq_close(msgq_id);
    
        
    return 0;