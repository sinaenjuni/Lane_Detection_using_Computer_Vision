#include "hough_transform.hpp"

cv::Mat calcGrayHist(const cv::Mat& img){
    cv::Mat hist;
    int channels[] = {0};
    int dims = 1;
    const int hist_size[] = { 256 };
    float level[] = {0,256};
    const float* ranges[] = {level};

    calcHist(&img, 1, channels, cv::Mat(), hist, dims, hist_size, ranges);
    return hist;
}

cv::Mat getGrayHistImage(const cv::Mat& hist)
{
	double histMax = 0.;
	minMaxLoc(hist, 0, &histMax);

	cv::Mat imgHist(100, 256+20, CV_8UC3, cv::Scalar(255,255,255));
	for (int i = 0; i < 256; i++) {
		line(imgHist, cv::Point(i+10, 100),
			cv::Point(i+10, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), cv::Scalar(0,0,0));
	}
    line(imgHist, cv::Point(128+10, 100),
			cv::Point(128+10, 0), cv::Scalar(255,0,0));
	return imgHist;
}

double calcMedian(cv::Mat src){
    std::vector<double> src_vec;
    src = src.reshape(0,1);
    src.copyTo(src_vec, cv::noArray());
    std::nth_element(src_vec.begin(), src_vec.begin() + src_vec.size() / 2, src_vec.end());
    return src_vec[src_vec.size() / 2];
}

cv::Mat canny(cv::Mat src, float sigma=0.33){
    double median = calcMedian(src);
    int32_t lower;
    int32_t upper;

    lower = (int)std::max(0.0  , (1.0-sigma) * median);
    upper = (int)std::min(255.0, (1.0+sigma) * median);
    cv::Canny(src, src, lower, upper);

    return src;
}

void histogram_stretching_mod(const cv::Mat& src, cv::Mat& dst){
    cv::Mat hist = calcGrayHist(src);
    
    int gmin=255, gmax=0;
    int ratio = int(src.cols * src.rows * 0.01);

    for(int i=0,s=0; i<255; i++){
        s+=(int)hist.at<float>(i);
        if(s>ratio){
            gmin=(int)i;
            break;
        }
    }

    for(int i=255,s=0; i>=0; i--){
        s+=(int)hist.at<float>(i);
        if(s>ratio){
            gmax=(int)i;
            break;
        }
    }
    dst = (src - gmin) * 255 / (gmax - gmin);

}



namespace My {

// Hough_lane_detection::Hough_lane_detection(std::string path){
//     mCap = cv::VideoCapture(path);

//     mImageWidth = mCap.get(cv::CAP_PROP_FRAME_WIDTH);
//     mImageHeight = mCap.get(cv::CAP_PROP_FRAME_HEIGHT);
//     std::cout << mImageHeight << " X " << mImageWidth << std::endl;
//     mROIStartHeight = 345;
//     mROIHeight = 40;
//     mRoi = cv::Rect(cv::Point(0, mROIStartHeight), cv::Point(mImageWidth, mROIStartHeight+mROIHeight));
    
//     mYAxisMargin = 10;

// }
template <typename PREC>
void Hough_lane_detection<PREC>::setParameters(std::string path){
    mCap = cv::VideoCapture(path);

    mImageWidth = mCap.get(cv::CAP_PROP_FRAME_WIDTH);
    mImageHeight = mCap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << mImageHeight << " X " << mImageWidth << std::endl;
    mROIStartHeight = 345;
    mROIHeight = 40;
    mRoi = cv::Rect(cv::Point(0, mROIStartHeight), cv::Point(mImageWidth, mROIStartHeight+mROIHeight));
    
    mYAxisMargin = 10;

}
template <typename PREC>
cv::Mat Hough_lane_detection<PREC>::preProcessing(cv::Mat src)
{
    static constexpr int32_t img_thr = 100;
    static constexpr int32_t img_thr_max = 255;
    static constexpr int32_t gaussian_filter = 3;
    static constexpr int32_t gaussian_sigma = 1;
    static constexpr int32_t mean_brightness = 128;
    std::vector<cv::Mat> plane;
    int m;

    // cv::Mat hist, src, dst, bin, roi;
    // cv::TickMeter tt;

    // cv::GaussianBlur(mFrame, mFrame, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma, 0);
    // cv::add(mFrame, (mean_brightness - cv::mean(mFrame, cv::noArray())[0]), mFrame);

    cv::cvtColor(src, src, cv::COLOR_BGR2HLS);
    cv::split(src, plane);

    src = plane[1];
    // roi = src(mRoi).clone();
    // dst = src.clone();
    // tt.start();
    cv::blur(src, src, cv::Size(3, 3));

    m = cv::mean(src)[0];
    src = src + (128 - m);
    histogram_stretching_mod(src, src);
    // cv::blur(src, src, cv::Size(3, 3));
    // cv::GaussianBlur(src, src, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma, gaussian_sigma);

    // int32_t gmin = src_m - 100, gmax = src_m + 50;
    // int32_t gmin = 50, gmax = 160;
    // src = (src - gmin) * 255 / (gmax - gmin);
    cv::dilate(src, src, cv::Mat(), cv::Point(-1,-1), 3);
    cv::erode(src, src, cv::Mat(), cv::Point(-1,-1), 3);

    // cv::Mat hist = getGrayHistImage(calcGrayHist(src));
    // cv::imshow("hist", hist);

    cv::threshold(src, src, 80, 255, cv::THRESH_BINARY_INV);
    // cv::dilate(roi, roi, cv::Mat(), cv::Point(-1,-1), 1);

    // cv::inRange(roi, cv::Scalar(0), cv::Scalar(60), roi);
    // roi=canny(roi);

    return src;
}

template <typename PREC>
std::pair<Indices, Indices> Hough_lane_detection<PREC>::divideLines(const Lines& lines)
{
    Indices leftLineIndices;
    Indices rightLineIndices;

    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    PREC slope;

    cv::Vec4f l;
    for (size_t i = 0; i < lines.size(); i++) {
        l = lines[i];
        
        // 0 1 2 3 x1 y1 x2 y2
        x1 = l[HoughIndex::x1]; 
        y1 = l[HoughIndex::y1];
        // point of start(top)
        x2 = l[HoughIndex::x2]; 
        y2 = l[HoughIndex::y2];
        // point of end(bottom)

        if(y1 > y2){
            std::swap(y1, y2);
            std::swap(x1, x2);
        }

        // 시작점 필터링
        if (y2 < mROIHeight - mYAxisMargin) continue;

        if(x1 - x2 == 0 || y1 - y2 == 0){
            slope = 0;
        }else{
            slope = static_cast<PREC>(y2 - y1) / static_cast<PREC>(x2 - x1);
        } 

        // if(slope == 0 || abs(slope) < 0.2) continue; 
        if ((fabsf(slope) < std::numeric_limits<PREC>::epsilon()) || abs(slope) < 0.2) continue; 

        // 평행선 filtering
        if(x2 < (mImageWidth/2)){
            // case is left
            leftLineIndices.push_back(i);
        }else{
            // case is right
            rightLineIndices.push_back(i);
        }
    }
            
    return { leftLineIndices, rightLineIndices };
}

template <typename PREC>
void Hough_lane_detection<PREC>::run(){
    static constexpr int32_t hough_rho = 1;
    static constexpr int32_t hough_min = 20;
    static constexpr int32_t hough_max_gab = 0;

    cv::Mat frame, frame_roi;
    Lines lines;

    mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);

    while (true)
    {   
        mCap >> frame;
        if (frame.empty()) continue;

        // std::cout << mCap.get(cv::CAP_PROP_POS_FRAMES) << "\n";
        frame_roi = frame(mRoi).clone();
        frame_roi = preProcessing(frame_roi);
        // hough
        cv::HoughLinesP(frame_roi, lines, hough_rho, CV_PI / 180, hough_min, hough_max_gab, mROIHeight-mYAxisMargin);
        cv::cvtColor(frame_roi, frame_roi, cv::COLOR_GRAY2BGR);


        // auto [hLeftLineIndices, hRightLineIndices] = divideLines(lines);
        // leftPositionX = getLinePositionX(lines, hLeftLineIndices, Direction::LEFT);
        // rightPositionX = getLinePositionX(lines, hRightLineIndices, Direction::RIGHT);

        cv::imshow("frame", frame);
        cv::imshow("roi", frame_roi);
        
        if(cv::waitKey(1) == 27) break;
        if(mCap.get(cv::CAP_PROP_POS_FRAMES) == 1800){
            mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);
        }
    }

    mCap.release();
    cv::destroyAllWindows();
}


}