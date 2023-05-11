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
        
        Lines mLines;
        std::pair<Indices, Indices> mDivideLines;
        int32_t mYAxisMargin;

    public:
        Hough_lane_detection(std::string path); 
        void run();
        cv::Mat preProcessing(cv::Mat src);
        std::pair<Indices, Indices> divideLines(const Lines& lines);
        int32_t getLinePositionX(const Lines& lines, const Indices& lineIndices, bool direction);
        std::pair<float, float> getLineParameters(const Lines& lines, const Indices& lineIndices);
        void getTrackPointX(int32_t positionX, int32_t& trackPosX, float areaMin, float areaMax);

};
