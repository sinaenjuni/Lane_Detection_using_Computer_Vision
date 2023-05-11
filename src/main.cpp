#include <iostream>
#include "hough_transform.hpp"

using PREC = float;

int main(int, char**) {
    // std::string path = "../data/Sub_project.avi";
    std::string path = "../data/20230510.mp4";
    std::cout << path << std::endl;


    // My::Hough_lane_detection<PREC> hough(path);
    My::Hough_lane_detection<PREC> hough;
    hough.setParameters(path);
    // hough.run();


}
