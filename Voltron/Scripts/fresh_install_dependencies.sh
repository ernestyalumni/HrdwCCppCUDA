#!/bin/bash
#
# REV:    0.1.A (Valid are A, B, D, T, Q, and P)
#               (For Alpha, Beta, Dev, Test, QA, and Production)
#
# PLATFORM: Not Platform Dependent (mostly Linux)
#
# PURPOSE: On fresh install, install all necessary packages.
#
# set -n   # Uncomment to check script syntax, without execution.
#          # NOTE: Do not forget to put the # comment back in or
#          #       the shell script will never execute!
# set -x   # Uncomment to debug this shell script
#
##########################################################
#         DEFINE FILES AND VARIABLES HERE
##########################################################

# cf. https://askubuntu.com/questions/459402/how-to-know-if-the-running-platform-is-ubuntu-or-centos-with-help-of-a-bash-scri

os_type=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

##########################################################
#              DEFINE FUNCTIONS HERE
# cf. Mastering Unix Shell Scripting, 2nd. Ed. Randal Michael. pp. 4-5.
# Functions. A function has the following form:
# function function_name
# {
#   commands to execute
# }
# or
# function_name ()
# {
#   commands to execute
# }
#
##########################################################

install_packages ()
{
  # Addition of more generally known expressions, operators "&&", "||" using
  # double brackets.
  if [[ "$1"="Pop!_OS$" || "$1"="Ubuntu" ]]
  then

    # Setup up necessary apt repositories or libraries to build Voltron; in
    # particular, libtbb-dev libboost-all-dev are both very necessary for cmake.

    sudo apt-get install build-essential ccache cmake libboost-all-dev \
      libtbb-dev

  elif [[ "$1"="Fedora Linux" ]]
  then
    sudo dnf install ccache cmake gcc-c++ tbb-devel boost-devel
  fi
}

install_optional_packages ()
{
  if [[ "$1"="Pop!_OS$" || "$1"="Ubuntu" ]]
  then

    sudo apt-get install valgrind
  fi
}

##########################################################
#               BEGINNING OF MAIN
##########################################################

echo $os_type

install_packages $os_type

# End of script
