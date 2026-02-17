# Stage 1: Build the binary
FROM rust:1.93.1-slim-bookworm as builder

# Create a workspace directory
WORKDIR /usr/src/app

# Copy the entire repo (or just the rust folder)
COPY ./rust /usr/src/app/rust

# Move into the rust directory and build
WORKDIR /usr/src/app/rust
RUN cargo build --release

# Stage 2: Create the tiny runtime image
FROM debian:bookworm-slim

# Copy the binary from the builder stage
COPY --from=builder /usr/src/app/rust/target/release/stitch /usr/local/bin/stitch

# Set the command to run your app
ENTRYPOINT ["stitch"]