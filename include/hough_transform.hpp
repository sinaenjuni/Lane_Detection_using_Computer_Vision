#include "opencv2/opencv.hpp"

using Line = cv::Vec4i;               ///< Line between two points
using Lines = std::vector<Line>;      ///< Vector of Lines
using Indices = std::vector<int32_t>; ///< Indices of lines



class Hough_lane_detection{
    private:
        cv::VideoCapture mCap;

        int32_t mImageWidth, mImageHeight;
        int32_t mROIStartHeight, mROIHeight, mMargin;
        cv::Rect mRoi;
        
        cv::Mat mFrame, mFrame_roi;
        std::vector<cv::Mat> mPlanes;
        Lines mLines;
        std::pair<Indices, Indices> mDivideLines;

    public:
        Hough_lane_detection(std::string path); 
        void run();
        void preProcessing();
        std::pair<Indices, Indices> divideLines(const Lines& lines);
        std::pair<int32_t, int32_t> getLanePosition();


};
