cf. https://doc.rust-lang.org/stable/book/ch01-03-hello-cargo.html

```
cargo new hello_cargo
cd hello_cargo
```
The first command creates a new directory and project called `hello_cargo`.

Git files won't be generated if you run `cargo new` within an existing Git repository. You can override this behavior using 
```
cargo new --vcs=git
```

`Cargo.toml` is Cargo's configuration format, TOML - Tom's Obvious, Minimal Language.
