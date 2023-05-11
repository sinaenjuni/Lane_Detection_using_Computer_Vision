cmake_minimum_required(VERSION 3.0.0)
project(Lane_detection VERSION 0.1.0)

# CPP 버전 지정!!
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

add_executable(Lane_detection src/main.cpp)
add_subdirectory(modules)

target_link_libraries(Lane_detection hough_transform)

include_directories(include)



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
