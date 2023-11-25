//-------------------------------------------------------------------------------------------------
/// \ref https://doc.rust-lang.org/stable/book/ch07-02-defining-modules-to-control-scope-and-privacy.html
/// 7.2 Defining Modules to Control Scope and Privacy.
/// \details When compiling a crate, compiler first looks in the crate root file (usually
/// src/lib.rs for library crate, src/main.rs for binary crate) for code to compile.
/// In crate root file, declare new modules.
/// Compiler will look for module's code in these places:
/// * inline, within curly brackets taht replace semicolon following mod module_name
/// * In file src/module_name.rs
/// * In file src/module_name/mod.rs
/// In an file other than crate root, you can declare submodules.
//-------------------------------------------------------------------------------------------------

// Code within a module is private from its parent modules by default. To make a module public,
// declare it with `pub mod` instead of `mod`.

pub mod data_structures;