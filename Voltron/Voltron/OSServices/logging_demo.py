"""
@file logging_demo.py

@details

@ref Logging HOWTO, https://docs.python.org/3/howto/logging.html
"""

"""
@brief Basic Logging Tutorial

@details Logging is a means of tracking events that happen when software runs.

Developer adds logging calls to their code to indicate certain events have
occurred.

An event is described by a descriptive message.
Descriptive message can optionally contain variable data (i.e. data potentially
different for each event occurrence).
Events also have an importance which the developer ascribes to the event; the
importance can also be called the level or severity.

When to use logging
@ref https://docs.python.org/3/howto/logging.html#when-to-use-logging

Task you want to perform; The best tool for the task

Display console output for ordinary usage of command line script or program;
`print()`

Report events occurring during normal operation of program (e.g. for status
monitoring or fault investigation);
`logging.info()` (or `logging.debug()` for very detailed output for diagnostic
purposes)

Issue warning regarding particular runtime event;
`warnings.warn()` in library code if issue is avoidable, and client application
should be modified to eliminate warning,
`logging.warning()` if there's nothing client application can do about
situation, but event should still be noted.

Report suppression of an error without raising an exception (.e.g error handler
in long-running server process);
`logging.error(), logging.exception()` or `logging.critical()` as appropriate
for specific error and application domain.


Standard Level or Severity of Events
Level; When it's used

'DEBUG'; Detailed info, typically of interest only when diagnosing problems.

'INFO'; Confirmation things working as expected.

'WARNING'; Indication something unexpected happened, or indicative of some
problem in the near future (e.g. "disk space low"). Software still working as
expected.

'ERROR'; Due to more serious problem, software not able to perform some
function.

'CRITICAL'; Serious error, indicating program itself may be unable to continue
running.

cf. https://realpython.com/python-logging/

"""
import logging

from imp import reload


if __name__ == "__main__":
    print("\nLogging Demo\n")

    # A very simple example.

    logging.warning('Watch out!') # will print a message to console
    # prints out WARNING:root:Watch out!
    # Printed message includes indication of level and description of the event
    # provided in logging call, i.e. "Watch out!"

    # Does not print out anything because default level is WARNING
    logging.info('I told you so') # will not print anything

    # https://stackoverflow.com/questions/31169540/python-logging-not-saving-to-file
    reload(logging)

    # Logging to a file

    # A very common situation is recording logging events in a file.
    
    # We set threshold to 'DEBUG', so all of the messages were printed (into
    # the file). So this shows how you can set logging level which acts as
    # the threshold for tracking.
    logging.basicConfig(
        filename='example_logging.log',
        # https://stackoverflow.com/questions/10706547/add-encoding-parameter-to-logging-basicconfig
        #encoding='utf-8',
        level=logging.DEBUG)
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')
    logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

    # logging level can be set from command-line option such as:
    # --log=INFO

    # And you have value of parameter passed for --log in some variable
    # 'loglevel', that you can use:

    # TODO: Write command line parsing to get --log into loglevel
    loglevel = "info"
    log_level_passed_in = getattr(logging, loglevel.upper())
    print("\n Log level passed in : ", log_level_passed_in, "\n")
