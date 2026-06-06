cmake_minimum_required(VERSION 3.10)
project(Project2)

set(CMAKE_CXX_STANDARD 17)

# tell CMake where to find OpenCV
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ../include)

# main image search program
add_executable(imgSearch imgSearch.cpp features.cpp csv_util.cpp filters.cpp)
target_link_libraries(imgSearch ${OpenCV_LIBS})