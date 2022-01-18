# syntax=docker/dockerfile:1
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:latest
SHELL ["/bin/bash", "-c"]

RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc && \
   apt clean && rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt-get -y install python3-venv
RUN pip3 install virtualenv
RUN git clone https://jungar111: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@github.com/jungar111/ml_ops_mnist.git
WORKDIR /ml_ops_mnist
ENV VIRUTALENV=env
RUN python3 -m venv ${VIRUTALENV}
ENV PATH="${VIRUTALENV}/bin:$PATH"
RUN make requirements
RUN dvc pull
RUN make data
CMD ["make", "train"]