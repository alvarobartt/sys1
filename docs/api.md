---
title: API
outline: false
---

# API

`sys1` exposes a JSON decision API compatible with the public
[TypeSafe AI System One HTTP format](https://docs.typesafe.ai/api). Requests
contain a `state` and named typed `questions`; responses contain structured
`answers` using Choice, Score, and Noul primitives. The examples below assume
the server is running at `http://localhost:3000`.

This reference is generated from the checked-in OpenAPI JSON on each docs
build. The build verifies that its version matches the crate. Endpoint methods,
request fields, content types, and response statuses come from that file.
Examples and deployment notes are maintained separately so they can add
context without duplicating the contract. Download the
[OpenAPI JSON](/openapi.json) for the complete machine-readable specification.
For a runnable first request, see [Get started](getting-started/index.md).

<!--@include: ./.vitepress/generated/api.md-->
