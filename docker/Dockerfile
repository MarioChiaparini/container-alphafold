ARG CUDA_FULL=11.2.2
FROM nvdia/cuda:${CUDA_FULL}-base-ubuntu20.4 AS build

ARG CUDA_FULL
ARG CUDA=11.2

ARG CUDA_JAXLIB=11.1
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash","-c"]

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
     build-essential \
     cmake \
     cuda-command-line-tools-${CUDA/./-} \
     git \
     hmmer \ 
     kalign \
     tzdata \
     wget \
   && rm -rf /var/lib/apt/lists
#CentOS systems
# Comment this next line if you don't use a Rocky linux or Fedora for example
# RUN yum update \
#    && yum upgrade -y \
#    && yum install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     cuda-command-line-tools-${CUDA/./-} \
#     git \
#     hmmer \
#     kalign \
#     tzdata \
#     wget \
#   && rm -rf /var/lib/apt/lists

RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      openmm=7.5.1 \
      python=3.8 \
      cudatoolkit==${CUDA_FULL} \
      pdbfixer \
      pip \
    && /opt/conda/bin/conda clean -aqy

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt


RUN pip3 install --upgrade --no-cache-dir pip \
    && pip3 install --no-cache-dir -r /app/alphafold/requirements.txt \
    && pip3 install --upgrade --no-cache-dir \
      jax==0.2.14 \
      jaxlib==0.1.69+cuda${CUDA_JAXLIB/./} \
      -f https://storage.googleapis.com/jax-releases/jax_releases.html


WORKDIR /opt/conda/lib/python3.8/site-packages
RUN patch -p0 < /app/alphafold/docker/openmm.patch

FROM nvidia/cuda:${CUDA_FULL}-base-ubuntu20.04
ARG OS=ubuntu2004
ARG CUDA=11.2

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      software-properties-common \
      cuda-command-line-tools-${CUDA/./-} \
      hmmer \
      kalign \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin \
    -O /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /" \
    && apt-get update \
    && apt-get install libcudnn8 libcusolver-${CUDA/./-} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/hhsuite /opt/hhsuite

RUN ln -s /opt/hhsuite/bin/* /usr/bin

RUN chmod u+s /sbin/ldconfig.real

WORKDIR /app/alphafold
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
  && chmod +x /app/run_alphafold.sh

ENV PATH=/opt/conda/bin:$PATH \
    LC_ALL=C.UTF-8 \
    ALPHAFOLD_BASE_DIR=/app/alphafold

ENTRYPOINT ["/app/run_alphafold.sh"]
