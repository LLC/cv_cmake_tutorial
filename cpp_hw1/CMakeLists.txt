# CMake 最小指定版本要求為3.10.2
cmake_minimum_required(VERSION 3.10.2)
# 設定 Project 名稱
project(cpphw1)
# 設定 C++ 版本為 11
# set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( $(OpenCV_INCLUDE_DIRS))
include_directories(include/)
aux_source_directory(./src DIR_SRCS)

add_executable( train1 main_train.cpp ${DIR_SRCS})
target_link_libraries( train1 ${OpenCV_LIBS} )
add_executable( test1 main_test.cpp ${DIR_SRCS})
target_link_libraries( test1 ${OpenCV_LIBS} )