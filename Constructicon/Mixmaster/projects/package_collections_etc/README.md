Created from `projects/` subdirectory by running

```
$ cargo new --lib package_collections_etc
```
This is how to create a package. cf. https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html

`Cargo.toml` gives us a package.

This package has 2 crates, a binary and library, because this package contains `src/main.rs` and `src/lib.rs`, "both with the same name as the package." (EY: ??)

The package has multiple binary crates by placing files in `src/bin` directory.

# Modules Cheat Sheet

cf. https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html

* **Start from the crate root**: 