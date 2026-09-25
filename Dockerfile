FROM rust:1.98.1-bookworm AS chef

ARG SCCACHE_VERSION=0.18.0
ARG SCCACHE_SHA256_AMD64=45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89
ARG SCCACHE_SHA256_ARM64=2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a
ARG TARGETARCH

RUN case "${TARGETARCH}" in \
        amd64) sccache_target=x86_64-unknown-linux-musl; sccache_sha256="${SCCACHE_SHA256_AMD64}" ;; \
        arm64) sccache_target=aarch64-unknown-linux-musl; sccache_sha256="${SCCACHE_SHA256_ARM64}" ;; \
        *) echo "unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl --proto '=https' --tlsv1.2 -fsSL \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${sccache_target}.tar.gz" \
        -o /tmp/sccache.tar.gz \
    && echo "${sccache_sha256}  /tmp/sccache.tar.gz" | sha256sum --check \
    && tar -xzf /tmp/sccache.tar.gz -C /tmp \
    && install "/tmp/sccache-v${SCCACHE_VERSION}-${sccache_target}/sccache" /usr/local/bin/sccache \
    && rm -rf /tmp/sccache.tar.gz "/tmp/sccache-v${SCCACHE_VERSION}-${sccache_target}"

RUN cargo install cargo-chef --version 0.1.78 --locked

ENV RUSTC_WRAPPER=/usr/local/bin/sccache

WORKDIR /app

FROM chef AS planner

COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY rust-toolchain.toml ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

COPY --from=planner /app/recipe.json recipe.json

ARG SCCACHE_GHA_ENABLED=off

RUN --mount=type=cache,id=sys1-sccache,target=/root/.cache/sccache,sharing=locked \
    --mount=type=secret,id=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=ACTIONS_RUNTIME_TOKEN \
    set -eu; \
    if [ "${SCCACHE_GHA_ENABLED}" = on ]; then \
        test -s /run/secrets/ACTIONS_RESULTS_URL; \
        test -s /run/secrets/ACTIONS_RUNTIME_TOKEN; \
        export SCCACHE_GHA_ENABLED=on; \
        export SCCACHE_GHA_VERSION=sys1-v1; \
        export ACTIONS_RESULTS_URL="$(cat /run/secrets/ACTIONS_RESULTS_URL)"; \
        export ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/ACTIONS_RUNTIME_TOKEN)"; \
    else \
        unset SCCACHE_GHA_ENABLED; \
    fi; \
    cargo chef cook --release --no-default-features --features cpu --recipe-path recipe.json; \
    sccache --show-stats

COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY rust-toolchain.toml ./
COPY src ./src

RUN --mount=type=cache,id=sys1-sccache,target=/root/.cache/sccache,sharing=locked \
    --mount=type=secret,id=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=ACTIONS_RUNTIME_TOKEN \
    set -eu; \
    if [ "${SCCACHE_GHA_ENABLED}" = on ]; then \
        test -s /run/secrets/ACTIONS_RESULTS_URL; \
        test -s /run/secrets/ACTIONS_RUNTIME_TOKEN; \
        export SCCACHE_GHA_ENABLED=on; \
        export SCCACHE_GHA_VERSION=sys1-v1; \
        export ACTIONS_RESULTS_URL="$(cat /run/secrets/ACTIONS_RESULTS_URL)"; \
        export ACTIONS_RUNTIME_TOKEN="$(cat /run/secrets/ACTIONS_RUNTIME_TOKEN)"; \
    else \
        unset SCCACHE_GHA_ENABLED; \
    fi; \
    cargo build --release --locked --no-default-features --features cpu --bin sys1; \
    sccache --show-stats

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 10001 --create-home sys1
ENV HF_HOME=/home/sys1/.cache/huggingface

COPY --from=builder /app/target/release/sys1 /usr/local/bin/sys1

USER sys1

EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/sys1"]
