#!/bin/bash

# This is normally first line of script, system still uses this line, instructs interpreter of system to execute script.
# If we don't add this line, then commands will be run within current shell; it may cause issues if we run another shell.
#!/bin/bash

# cf. Mastering Linux Shell Scripting

# Do chmod +x bash_script_notes.sh too

# $# is number of arguments. -gt is greater than.
if [ $# -gt 0 ]; then
  echo "$# number of arguments, Hello $1"
else
  echo "${#} number of arguments"
fi

# cf. Lecture 02 - Shell Scripting, 15-123S09 CS CMU

# Quotes

# unquoted strings are normally interpreted
# "quoted strings are basically literals -- but $variables are evaluated"
# 'quoted strings are absolutely literally interpreted'
# `commands in quotes like this are executed, their output is then inserted as
# if it were assigned to a variable and then that variable was evaluated`

