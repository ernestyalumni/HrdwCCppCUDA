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

# Expressions, Strings, pp. 92, Ch. 6 Expressions of
# Linux Shell Scripting with Bash, Ken. O. Burtch.
#
# -z s--(zero length) True if string is empty
# -n s (or just s) -- (not null) True if string is not empty
# s1 = s2 - True if string s1 is equal to s2
# s1 != S2
# s1 < s2
# s1 >> s2

# pp. 100, Ch. 6 Expressions
# Arithmetic Tests
# n1 -eq n2
# n1 -ne n2
# n1 -lt n2