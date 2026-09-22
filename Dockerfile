FROM rust:1.98.1-bookworm AS chef

RUN cargo install cargo-chef --version 0.1.78 --locked

WORKDIR /app

FROM chef AS planner

COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --no-default-features --features cpu --recipe-path recipe.json
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY src ./src
RUN cargo build --release --locked --no-default-features --features cpu --bin sys1

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home sys1
ENV HF_HOME=/home/sys1/.cache/huggingface

COPY --from=builder /app/target/release/sys1 /usr/local/bin/sys1

USER sys1

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/sys1"]
