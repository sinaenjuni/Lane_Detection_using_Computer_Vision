#include <iostream>
#include "hough_transform.hpp"

int main(int, char**) {
    // std::string path = "../data/Sub_project.avi";
    std::string path = "../data/20230510.mp4";
    std::cout << path << std::endl;


    Hough_lane_detection hough(path);
    hough.run();


}
