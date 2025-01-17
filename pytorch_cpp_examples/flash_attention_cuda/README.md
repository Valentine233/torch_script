# Flash Attention CUDA

This repository contains CPP examples of using Flash Attention on CUDA.

## Overview

Flash Attention is an efficient attention mechanism that reduces memory usage and improves performance. This repository provides examples of how to run CUDA Flash Attention with Pure CPP API.
Note that the repository already contains the built products in directory `build` and also the related building log `build.log`.

## Requirements

- CUDA supporting SM >= 80
- PyTorch

## Installation

1. Clone the required dependencies:
    ```bash
    # 
    git clone https://github.com/Dao-AILab/flash-attention # Flash-Attention REPO
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/Valentine233/torch_script
    cd pytorch_cpp_exmples/flash_attention_cuda
    ```

## Usage

1. Build and compile the CUDA code:
    ```bash
    ./build.sh
    ```

2. Run the example script:
    ```bash
    cd build
    ./flash_attention_cuda
    ```
