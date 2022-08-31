ARG BASE_IMAGE=nvidia/cuda:11.4.3-base-ubuntu20.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt apt-get update
# RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
      git \
      cmake \
      build-essential \
      libboost-program-options-dev \
      libboost-filesystem-dev \
      libboost-graph-dev \
      libboost-system-dev \
      libboost-test-dev \
      libeigen3-dev \
      libsuitesparse-dev \
      libfreeimage-dev \
      libmetis-dev \
      libgoogle-glog-dev \
      libgflags-dev \
      libglew-dev \
      qtbase5-dev \
      libqt5opengl5-dev \
      libcgal-dev \
      ca-certificates \
      ccache \
      curl \
      libjpeg-dev \
      libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# Build and install ceres solver
RUN apt-get update
RUN apt-get install -y libatlas-base-dev libsuitesparse-dev
ARG CERES_SOLVER_VERSION=2.1.0
RUN git clone https://github.com/ceres-solver/ceres-solver.git --tag ${CERES_SOLVER_VERSION}
RUN cd ${CERES_SOLVER_VERSION} && \
	mkdir build && \
	cd build && \
	cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
	make -j4 && \
	make install

# Build and install COLMAP

# Note: This Dockerfile has been tested using COLMAP pre-release 3.7.
# Later versions of COLMAP (which will be automatically cloned as default) may
# have problems using the environment described thus far. If you encounter
# problems and want to install the tested release, then uncomment the branch
# specification in the line below
RUN git clone https://github.com/colmap/colmap.git #--branch 3.7

RUN cd colmap && \
	git checkout dev && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j4 && \
	make install

# Build and install PyTorch
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt install -y pip

RUN pip install \
    numpy \
    matplotlib \
    torch \
    imageio \
    h5py \
    deepdish \
    torchvision \
    jupyterlab \
    notebook

# ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
CMD ["bash"]
EXPOSE 8888
