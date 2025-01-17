# rm -rf build
mkdir -p build
cd build
cmake -DCAFFE2_USE_CUDNN=True -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_VERBOSE_MAKEFILE=ON ..
make VERBOSE=1
