FROM rust:1.80 as builder
WORKDIR /app
COPY ../../rust ./
RUN cargo build --release -p service

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/service /usr/local/bin/service
EXPOSE 8080 50051
CMD ["/usr/local/bin/service"]
