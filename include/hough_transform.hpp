#include "opencv2/opencv.hpp"



namespace My
{
    using Line = cv::Vec4i;               ///< Line between two points
    using Lines = std::vector<Line>;      ///< Vector of Lines
    using Indices = std::vector<int32_t>; ///< Indices of lines


    enum HoughIndex : uint8_t
    {
        x1 = 0, ///< First point x
        y1 = 1, ///< First point y
        x2 = 2, ///< Second point x
        y2 = 3, ///< Second point y
    };


    template <typename PREC>
    class Hough_lane_detection
    {
    private:
        cv::VideoCapture mCap;

        int32_t mImageWidth, mImageHeight;
        int32_t mROIStartHeight, mROIHeight, mYAxisMargin;
        cv::Rect mRoi;
        
        std::pair<Indices, Indices> mDivideLines;

    public:
        // Hough_lane_detection(std::string path); 
        void setParameters(std::string path);
        void run();
        cv::Mat preProcessing(cv::Mat src);
        std::pair<Indices, Indices> divideLines(const Lines& lines);
        // std::pair<Indices, Indices> divideLines(const Lines& lines);
        // std::pair<int32_t, int32_t> getLanePosition();


};
} // namespace My
