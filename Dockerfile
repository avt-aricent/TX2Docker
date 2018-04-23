# This is the base container for the Jetson TX2 board with drivers (with cuda)
FROM arm64v8/ubuntu:xenial-20180123

# base URL for NVIDIA libs
ARG URL=http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/3.2/pwv346/JetPackL4T_32_b157

# Update packages, install some useful packages
RUN apt-get update && apt-get install -y \
	apt-utils \
	bzip2 \ 
	curl \
        sudo \
        unp \
    && apt-get clean \
    && rm -rf /var/cache/apt

WORKDIR /tmp

# Install drivers first
RUN curl -sL http://developer.nvidia.com/embedded/dlc/l4t-jetson-tx2-driver-package-28-2 | tar xvfj -
RUN chown root /etc/passwd /etc/sudoers /usr/lib/sudo/sudoers.so /etc/sudoers.d/README
RUN /tmp/Linux_for_Tegra/apply_binaries.sh -r / && rm -fr /tmp/*


## Pull the rest of the jetpack libs for cuda/cudnn and install
RUN curl $URL/cuda-repo-l4t-9-0-local_9.0.252-1_arm64.deb -so cuda-repo-l4t_arm64.deb
RUN curl $URL/libcudnn7_7.0.5.13-1+cuda9.0_arm64.deb -so /tmp/libcudnn_arm64.deb
RUN curl $URL/libcudnn7-dev_7.0.5.13-1+cuda9.0_arm64.deb -so /tmp/libcudnn-dev_arm64.deb

## Install libs: L4T, CUDA, cuDNN
RUN dpkg -i /tmp/cuda-repo-l4t_arm64.deb
RUN apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
RUN apt-get update && apt-get install -y cuda-toolkit-9.0
RUN dpkg -i /tmp/libcudnn_arm64.deb
RUN dpkg -i /tmp/libcudnn-dev_arm64.deb
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra

## Re-link libs in /usr/lib/<arch>/tegra
RUN ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.28.2.0 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so
RUN ln -s /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.28.2.0 /usr/lib/aarch64-linux-gnu/tegra/libnvidia-ptxjitcompiler.so.1
RUN ln -sf /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so
# D.R. -- need to do this for some strange reason (for jetson tx2)
RUN ln -s /usr/lib/aarch64-linux-gnu/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so.1

## Clean up (don't remove cuda libs... used by child containers)
RUN apt-get -y autoremove && apt-get -y autoclean
RUN rm -rf /var/cache/apt



# Install the necessary libraries for python 3.6
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:jonathonf/python-3.6

# Install python 3.6
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-pip\
    && rm -rf /var/lib/apt/lists/*

# Install the necessary libraries for graphviz
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install the necessary python libraries
RUN pip3 install \
    graphviz \
    && rm -r /root/.cache/pip

RUN pip3 install \
    jupyter \
    && rm -r /root/.cache/pip

# Install the necessary libraries for python 3.6
RUN apt-get update && apt-get install -y \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    scipy \
    && rm -r /root/.cache/pip

RUN pip3 install \
    keras \
    && rm -r /root/.cache/pip

# Install the necessary libraries for matplotlib
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libpng12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    matplotlib \
    && rm -r /root/.cache/pip

RUN pip3 install \
    numpy \
    && rm -r /root/.cache/pip

RUN pip3 install \
    pandas \
    && rm -r /root/.cache/pip

RUN pip3 install \
    pydot \
    && rm -r /root/.cache/pip


RUN pip3 install \
    seaborn \
    && rm -r /root/.cache/pip

RUN pip3 install \
    sklearn \
    && rm -r /root/.cache/pip

RUN pip3 install \
    statsmodels \
    && rm -r /root/.cache/pip

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    h5py \
    && rm -r /root/.cache/pip

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

#RUN wget https://github.com/jetsonhacks/installTensorFlowJetsonTX/tree/master/TX2/tensorflow-1.3.0-cp35-cp35m-linux_aarch64.whl

#RUN ls -al

#RUN pip3 install \
#    ./tensorflow-1.3.0-cp35-cp35m-linux_aarch64.whl \
#    && rm -r /root/.cache/pip




###########################################
# Install tensorflow


# installPrerequisitesPy3.sh
#./scripts/installDependenciesPy3.sh

# Install Java
RUN sudo add-apt-repository ppa:webupd8team/java
RUN sudo apt-get update
RUN sudo apt-get install oracle-java8-installer -y

# Install other dependencies
RUN sudo apt-get install zip unzip autoconf automake libtool curl zlib1g-dev maven -y

# Install Python 3.x
RUN sudo apt-get install python3-numpy swig python3-dev python3-pip python3-wheel -y

#./scripts/installBazel.sh
INSTALL_DIR=$PWD
cd $HOME
wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-dist.zip
unzip bazel-0.10.0-dist.zip -d bazel-0.10.0-dist
sudo chmod -R ug+rwx $HOME/bazel-0.10.0-dist
# git clone https://github.com/bazelbuild/bazel.git
cd bazel-0.10.0-dist
./compile.sh 
sudo cp output/bazel /usr/local/bin


# cloneTensorFlow.sh
INSTALL_DIR=$PWD
cd $HOME
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v1.3.0
patch -p1 < $HOME/patches/tensorflow.patch
# Patch up the Workspace.bzl for the Github Checksum issue
patch -p1 < $HOME/patches/workspacebzl.patch


# setTensorFlowEVPy3.sh

cd $HOME/tensorflow
# TensorFlow couldn't find include file for some reason
# TensorFlow expects it in /usr/lib/aarch64-linux-gnu/include/cudnn.h
sudo mkdir /usr/lib/aarch64-linux-gnu/include/
sudo cp /usr/include/cudnn.h /usr/lib/aarch64-linux-gnu/include/cudnn.h
# Setup the environment variables for configuration
# PYTHON Path is the default
default_python_bin_path=$(which python3)
PYTHON_BIN_PATH=$default_python_bin_path
# No Google Cloud Platform support
TF_NEED_GCP=0
# No Hadoop file system support
TF_NEED_HDFS=0
# Use CUDA
TF_NEED_CUDA=1
# Setup gcc ; just use the default
default_gcc_host_compiler_path=$(which gcc)
GCC_HOST_COMPILER_PATH=$default_gcc_host_compiler_path
# TF CUDA Version 
TF_CUDA_VERSION=9.0
# CUDA path
default_cuda_path=/usr/local/cuda
CUDA_TOOLKIT_PATH=$default_cuda_path
# cuDNN
TF_CUDNN_VERSION=7.0.5
default_cudnn_path=/usr/lib/aarch64-linux-gnu
CUDNN_INSTALL_PATH=$default_cudnn_path
# CUDA compute capability
TF_CUDA_COMPUTE_CAPABILITIES=6.2
CC_OPT_FLAGS=-march=native
TF_NEED_JEMALLOC=1
TF_NEED_OPENCL=0
TF_ENABLE_XLA=0
# Added for TensorFlow 1.3
TF_NEED_MKL=0
TF_NEED_MPI=0
TF_NEED_VERBS=0
# Use nvcc for CUDA compiler
TF_CUDA_CLANG=0


source ./configure


# buildTensorFlow.sh
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=9.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export TF_CUDNN_VERSION=7.0.5
export CUDNN_INSTALL_PATH=/usr/lib/aarch64-linux-gnu/
export TF_CUDA_COMPUTE_CAPABILITIES=6.2

# Build Tensorflow
cd $HOME/tensorflow
bazel build -c opt --local_resources 3072,4.0,1.0 --verbose_failures --config=cuda //tensorflow/tools/pip_package:build_pip_package



# packageTensorFlow.sh


# Install wheel file





















# Start jupyter notebook
CMD jupyter-notebook --ip="*" --no-browser --allow-root

