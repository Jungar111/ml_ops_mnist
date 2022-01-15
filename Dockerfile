# syntax=docker/dockerfile:1
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:latest

RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc && \
   apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://jungar111: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@github.com/jungar111/ml_ops_mnist.git
WORKDIR /ml_ops_mnist
RUN make requirements
COPY data/ data/
RUN make data
RUN make train