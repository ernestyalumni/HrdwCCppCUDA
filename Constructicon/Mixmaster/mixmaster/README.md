# Installing Rust

## Mac OS Apple Silicon Rust install

https://rust-lang.org/tools/install/

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

# Running "integration" tests

The separate tests subdirectory (`mixmaster/tests`) are considered "integration tests" in Rust, even if their simplicity makes them more "unit tests". Do

```
cargo test --test ./tests
```