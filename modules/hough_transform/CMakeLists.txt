cmake_minimum_required(VERSION 3.0.0)
project(hough_transform LANGUAGES CXX)

add_library(hough_transform hough_transform.cpp)

# find_package(OpenCV 4.7.0 EXACT REQUIRED HINTS ${CMAKE_SOURCE_DIR}/thirdparty/OpenCV/install/lib/cmake/opencv4)
find_package(OpenCV 4.7.0 EXACT REQUIRED HINTS ~/thirdparty/OpenCV/install/lib/cmake/opencv4)
if (OpenCV_FOUND)
    message(STATUS "OpenCV Found! - ${OpenCV_DIR}")
    message(STATUS "OpenCV Found! - ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Found! - ${OpenCV_LIBS}")
endif()
target_include_directories(hough_transform PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hough_transform PUBLIC ${OpenCV_LIBS})

