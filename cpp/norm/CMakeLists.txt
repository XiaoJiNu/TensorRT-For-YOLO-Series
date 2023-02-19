cmake_minimum_required(VERSION 2.6)

project(yolo)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)

include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1//lib64)

#
# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)


# tensorRT
# Add the given directories to those the compiler uses to search for include files.
# Relative paths are interpreted as relative to the current source directory
# 将给定目录添加到编译器搜索头文件的路径中
include_directories(/home/yr/software/TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.1.5/include/)
# ----- link_directories解释 -----
# Add directories in which the linker will look for libraries
# 添加⾮标准的共享库搜索路径，链接器会在其中搜索非标准的共享库
# LINK_DIRECTORIES 命令来指定第三方库所在路径，比如，你的动态库在/home/myproject/libs这个路径下，则通命令：
# LINK_DIRECTORIES(/home/myproject/libs)，把该路径添加到第三方库搜索路径中，***这样就可以使用相对路径了***，
# 使用TARGET_LINK_LIBRARIES的时候，只需要给出动态链接库的名字就行了。
# 也可以set(LINK_DIR /Users/haoran/Downloads/wfdb/lib)，LINK_DIRECTORIES({LINK_DIR})
# 官方不建议使用该命令，取而代之的为find_package()  find_library()
link_directories(/home/yr/software/TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.1.5/lib/)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolo ${PROJECT_SOURCE_DIR}/main.cpp yolo.hpp)
# ----- target_link_libraries解释 -----
# 该指令的作用为将目标文件与库文件进行链接。
# yr: 链接器在库搜索路径中去搜索。库搜索路径在前面的link_directories给出了
# 问题： 库搜索路径除了前面的link_directories给出，还有其他路径吗？
target_link_libraries(yolo nvinfer)
target_link_libraries(yolo cudart)
target_link_libraries(yolo ${OpenCV_LIBS})

add_definitions(-O2 -pthread)