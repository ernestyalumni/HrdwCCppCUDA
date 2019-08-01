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

### ['add_library'](https://cmake.org/cmake/help/latest/command/add_library.html)

Add a library to the project using specified source files.

#### What is a library?

**What is a library?**

cf. [A.1 Static and dynamic libraries](https://www.learncpp.com/cpp-tutorial/a1-static-and-dynamic-libraries/)

A **library** is a package of code that's meant to be reused by many programs. Typically, a *C++ library* comes in 2 pieces:

1. header file that defines the functionality the library is exposing to programs using it
2. A precompiled binary that contains the implementation, pre-compiled into machine language.

Libraries are precompiled for several reasons,
- since libraries rarely change, they don't need to be recompiled often
- because precompiled objects are in machine language, prevents people from accessing or changing the source code, 

there's 2 types, static and dynamic:

- **static library** (a.k.a. **archive**) consists of routines that are compiled and linked directly into your program. When you compile a program that uses a static library, all functionality of the static library that your program uses becomes part of your executable.
  * in Windows, static libraries typically have `.lib` extension, linux, `.a` (archive) extension
  * 1 advantage: you only have to distribute executable in order for users to run your program.
- **dynamic library** (a.k.a. **shared library**) consists of routines that are loaded into application at runtime. 
  * When you compile a program that uses a dynmaic library, library doesn't become part of executable - it remains as a separate unit.
  * On Windows, dynamic libraries typically have `.dll` (dynamic link library) extension, linux `.so` (shared object) extension
  * 1 advantage, is that many programs can shared 1 copy, which saves space. Perhaps a bigger advantage is that dynamic library can be upgraded to a newer version without replacing all executables that use it.


#### Normal Libraries

```
add_library(<name> [STATIC | SHARED | MODULE]
  [EXCLUDE_FROM_ALL]
  [source1] [source2 ...])
```

Adds a library target called `<names>` to be built from source files listed in the command invocation.  (Source files can be omitted if they're added later using `target_sources()`.) 

- `STATIC` - `STATIC` libraries are archives of object files for use when linking other targets.
- `SHARED` libraries are linked dynamically and loaded at runtime.
- `MODULE` - libraries are plugins that aren't linked into other targets but maybe loaded dynamically at runtime using dlopen-like functionality.

cf. https://cliutils.gitlab.io/modern-cmake/chapters/basics.html

If you leave this choice off, `BUILD_SHARED_LIBS` will be used to pick between `STATIC` and `SHARED`.


### ['add_subdirectory'](https://cmake.org/cmake/help/v3.0/command/add_subdirectory.html)

Add a subdirectory to the build.
```
add_subdirectory(source_dir [binary_dir]
  [EXCLUDE_FROM_ALL])
```

The `source_dir` specifies directory in which the source `CMakeLists.txt` and code files are located. If it's a relative path, it'll be evaluated with respect to the current directory (typical usage), but may also be an absolute path.

`binary_dir` specifies directory in which to place output files. If `binary_dir` isn't specified, value of `source_dir`, before expanding any relative path, will be used (typical usage).

The `CMakelists.txt` file in the specified source directory will be processed immediately by CMake before processing in the current input file continues beyond this command.

If `EXCLUDE_FROM_ALL` argument is provided, then targets in subdirectory won't be included in the `ALL` target of parent directory by default, and will be excluded from IDE project files.




### [`CheckCXXCompilerFlag`](https://cmake.org/cmake/help/v3.14/module/CheckCXXCompilerFlag.html)

Check whether the CXX compiler supports a given flag.

** `check_cxx_compiler_flag`

https://cmake.org/cmake/help/v3.15/prop_tgt/LANG_COMPILER_LAUNCHER.html?highlight=compiler

`<LANG>_COMPILER_LAUNCHER`, 

This property is implemented only when `<LANG>` is `C`, `CXX`, or `CUDA`, e.g.

`CUDA_COMPILER_LAUNCHER`, `CXX_COMPILER_LAUNCHER`.

Specify a semicolon-separated list containing a command line for a compiler launching tool. The Makefile Generators and the Ninja generator will run this tool and pass the compiler and its arguments to the tool. e.g. Example tools are distcc and ccache.

This property is *initialized* by the value of the `CMAKE_<LANG>_COMPILER_LAUNCHER` variable if it's set when a target is created.


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


## Abridged explanations of `cmake-modules`

### FindBoost

Find Boost include dirs and libraries.

Use this module by invoking `find_packages` with the form:

```
find_package(Boost
  [version] [EXACT]
  [REQUIRED]
  [COMPONENTS <libs>...] # Boost libraries by their canonical name
  ) # e.g. "date_time" for "libboost_date_time"
```