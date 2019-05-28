# CMake

cf. https://cmake.org/cmake-tutorial/

## Command Line

### Typical usage

In the "Source" directory with the "top-level" `CMakeLists.txt`, first, you'll want to "get out of it," make a separate build directory, and proceed ...

```
cd ..
mkdir -p BuildGcc # e.g. Gcc could be Clang, if you want to use Clang 
cd BuildGcc
```
```
cmake -DCMAKE_CXX_COMPILER=g++ ../Source
```
(Just follow how to ["Generate a Project Buildsystem"](https://cmake.org/cmake/help/latest/manual/cmake.1.html#generate-a-project-buildsystem))

### [`cmake(1)`](https://cmake.org/cmake/help/latest/manual/cmake.1.html)

```
# Generate a Project Buildsystem
cmake [<options>] <path-to-source>
cmake [<options>] <path-to-existing-build>
cmake [<options>] -S <path-to-source> -B <path-to-build>

# Build a Project

cmake --build <dir> [<options>] [-- <build-tool-options>]
```

The **cmake** exectuable is the command-line interface of the cross-platform buildsystem generator CMake.

#### What is a CMake Buildsystem?

*buildsystem* - describes how to build a project's executables and libraries from its source code, using a *build tool* to automate the process. e.g., 
- buildsystem may be a `Makefile` for use with a command-line `make` tool.

In order to avoid maintaining multiple such buildsystems, a project may specifiy its buildsystem abstractly using files written in `CMake language`.

#### [Generate a Project Build system](https://cmake.org/cmake/help/latest/manual/cmake.1.html#generate-a-project-buildsystem)

Run CMake with 1 of the following command signatures to specify the source and build trees, and generate a buildsystem:

```
cmake [<options>] <path-to-source>
```

Uses current working directory as the build tree, and `<path-to-source>` as source tree. Specified path may be absolute or relative to the current working directory. 
  - Source tree (i.e. directory) must contain a `CMakeLists.txt` file and must *not* contain a `CMakeCache.txt` file because latter identifies an existing build tree. 



## Abridged explanations of `cmake-commands`

### [`CheckCXXCompilerFlag`](https://cmake.org/cmake/help/v3.14/module/CheckCXXCompilerFlag.html)

Check whether the CXX compiler supports a given flag.

**`check_cxx_compiler_flag`

```
check_cxx_compiler_flag(<flag> <var>)
```
Check that `<flag>` is accepted by compiler without a diagnostic. Stores result in an internal cache entry named `<var>`. 

e.g. cf. [stack overflow "Getting CMake `CHECK_CXX_COMPILER_FLAG` to work."](https://stackoverflow.com/questions/25451254/getting-cmake-check-cxx-compiler-flag-to-work)

```
# Add the c++14 flag, whatever it is.
include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG(-std=c++0x COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
```

### [`include`](https://cmake.org/cmake/help/v3.14/command/include.html)

Load and run CMake code from a file or module.
```
include (<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
  [NO_POLICY_SCOPE])
```

### include_directories

### ['add_library']()



### [`project`](https://cmake.org/cmake/help/v3.6/command/project.html#command:project)

Set a name, version, and enable languages for entire project.

```
project(<PROJECT-NAME> [LANGUAGES] [<language-name>...])
project(<PROJECT-NAME)>
  [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
  [LANGUAGES <language-name>...])
```
Sets name of project and stores name in [`PROJECT_NAME`](https://cmake.org/cmake/help/v3.6/variable/PROJECT_NAME.html#variable:PROJECT_NAME) variable. Also, sets variables
* [`PROJECT_SOURCE_DIR`](https://cmake.org/cmake/help/v3.6/variable/PROJECT_SOURCE_DIR.html#variable:PROJECT_SOURCE_DIR), [`<PROJECT-NAME>_SOURCE_DIR`](https://cmake.org/cmake/help/v3.6/variable/PROJECT-NAME_SOURCE_DIR.html#variable:%3CPROJECT-NAME%3E_SOURCE_DIR)
* `PROJECT_BINARY_DIR`, `<PROJECT-NAME>_BINARY_DIR`

### [`PROJECT_SOURCE_DIR`](https://cmake.org/cmake/help/v3.3/variable/PROJECT_SOURCE_DIR.html)

Top level source directory for current project; this is the source directory of the most recent `project()` command, i.e. 

`PROJECT_SOURCE_DIR` refers to folder of the `CMakeLists.txt` containing the most recent `project()`, so for 

```
project(Voltron)
```
`PROJECT_SOURCE_DIR` for `Voltron` is the same directory as the `CMakeLists.txt`.

### [`set`](https://cmake.org/cmake/help/latest/command/set.html)

Set a normal, cache, or environment variable to a given value.


#### Set Normal Variable
```
set(<variable> <value>... [PARENT_SCOPE])
```
Sets the given `<variable>` in the current function or directory scope.

e.g.
```
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
```

cf. [stackoverflow, "Inducing minimal C++ standard version in CMake"](https://stackoverflow.com/questions/48148275/inducing-minimal-c-standard-version-in-cmake)