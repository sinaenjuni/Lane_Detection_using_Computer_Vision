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


void histogram_stretching_mod(const cv::Mat& src, cv::Mat& dst){
    cv::Mat hist = calcGrayHist(src);;
    
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

Hough_lane_detection::Hough_lane_detection(std::string path){
    mCap = cv::VideoCapture(path);

    mImageWidth = mCap.get(cv::CAP_PROP_FRAME_WIDTH);
    mImageHeight = mCap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << mImageHeight << " X " << mImageWidth << std::endl;
    mROIStartHeight = 345;
    mROIHeight = 40;
    mRoi = cv::Rect(cv::Point(0, mROIStartHeight), cv::Point(mImageWidth, mROIStartHeight+mROIHeight));
    
    mMargin = 10;

}

void Hough_lane_detection::preProcessing(){
    static constexpr int32_t img_thr = 100;
    static constexpr int32_t img_thr_max = 255;
    static constexpr int32_t gaussian_filter = 3;
    static constexpr int32_t gaussian_sigma = 1;
    static constexpr int32_t mean_brightness = 128;

    cv::Mat hist, src, dst, bin, roi;
    cv::TickMeter tt;

    // cv::GaussianBlur(mFrame, mFrame, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma, 0);
    // cv::add(mFrame, (mean_brightness - cv::mean(mFrame, cv::noArray())[0]), mFrame);

    cv::cvtColor(mFrame, mFrame_roi, cv::COLOR_BGR2HLS);
    cv::split(mFrame_roi, mPlanes);

    src = mPlanes[1];
    roi = src(mRoi).clone();
    // dst = src.clone();
    tt.start();
    cv::blur(roi, roi, cv::Size(3, 3));

    int roi_m = cv::mean(roi)[0];
    roi = roi + (128 - roi_m);
    histogram_stretching_mod(roi, roi);
    // cv::blur(roi, roi, cv::Size(3, 3));
    // cv::GaussianBlur(roi, roi, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma, gaussian_sigma);

    // int32_t gmin = roi_m - 100, gmax = roi_m + 50;
    // int32_t gmin = 50, gmax = 160;
    // roi = (roi - gmin) * 255 / (gmax - gmin);
    cv::dilate(roi, roi, cv::Mat(), cv::Point(-1,-1), 3);
    cv::erode(roi, roi, cv::Mat(), cv::Point(-1,-1), 3);


    hist = getGrayHistImage(calcGrayHist(src));
    cv::imshow("srchist", hist);
    hist = getGrayHistImage(calcGrayHist(roi));
    cv::imshow("roihist", hist);

    // cv::threshold(roi, roi, 80, 255, cv::THRESH_BINARY_INV);
    // cv::dilate(roi, roi, cv::Mat(), cv::Point(-1,-1), 1);

    // cv::inRange(roi, cv::Scalar(0), cv::Scalar(60), roi);
    // roi=canny(roi);
    
    
    tt.stop();
    std::cout << tt.getTimeMilli()  << "\n";

    // tt.reset();
    // tt.start();
    // int src_m = cv::mean(src)[0];

    // gmin = 0, gmax = 255;
    // src = (src - gmin) * 255 / (gmax - gmin);
    // tt.stop();
    // double src_tt = tt.getTimeMilli();

    // std::cout << roi_tt << " " << src_tt << "\n";
    // std::cout << roi_m << " " << src_m << "\n";

    // cv::blur(dst, dst, cv::Size(3, 3));

    // dst = dst + (128 - m);

    // cv::GaussianBlur(dst, dst, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma);

    // int32_t gmin = 0, gmax = 255;
    // dst = (dst - gmin) * 255 / (gmax - gmin);
    // cv::dilate(dst, dst, cv::Mat(), cv::Point(-1,-1), 1);

    // cv::erode(dst, dst, cv::Mat(), cv::Point(-1,-1), 1);

    // cv::GaussianBlur(dst, dst, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma);

    // histogram_stretching_mod(src, dst);

    // std::cout << m << "\n";
	// float alpha = 3.0f;
	// cv::Mat dst = mPlanes[2] + (mPlanes[2] - m) * alpha;
    // dst = src - 100;
    // dst = dst + 100;

    // double histMax = 0., histMin;
	// minMaxLoc(dst, &histMin, &histMax); 
    // std::cout << histMin << " " << histMax << "\n";


    // cv::threshold(mPlanes[2], mPlanes[2], 180, 255, cv::THRESH_BINARY);
    // dst.copyTo(dst, (dst > 50));

    cv::imshow("src", src);
    cv::imshow("dst", roi);



    // cv::inRange(dst, cv::Scalar(0), cv::Scalar(90), dst);
    // cv::erode(dst, dst, cv::Mat(), cv::Point(-1,-1), 1);

    // cv::threshold(dst, dst, 60, 255, cv::THRESH_BINARY_INV);
    // cv::imshow("bin", dst);



    // float32_t alpha = 1.0;
    // mFrame_roi = (1 + alpha) * mFrame_roi - 128 * alpha;


    // cv::imshow("H", mPlanes[0]);
    // cv::imshow("S", mPlanes[1]);
    // cv::imshow("V", mPlanes[2]);


    // cv::cvtColor(mFrame, mFrame_roi, cv::COLOR_BGR2Lab);
    // cv::split(mFrame_roi, mPlanes);
    
    // cv::imshow("L", mPlanes[0]);
    // cv::imshow("A", mPlanes[1]);
    // cv::imshow("B", mPlanes[2]);


    // mFrame_roi = mPlanes[2];

    // ROI crop
    // mFrame_roi = mFrame_roi(mRoi);

    // blur
    // cv::GaussianBlur(mFrame_roi, mFrame_roi, cv::Size(gaussian_filter, gaussian_filter), gaussian_sigma);

    // bright filter
    // cv::add(mFrame_roi, (mean_brightness - cv::mean(mFrame_roi, cv::noArray())[0]), mFrame_roi);

    // binarization
    // cv::threshold(mFrame_roi, mFrame_roi, img_thr, img_thr_max, cv::THRESH_BINARY_INV);

    // edge    
    // dst=canny(dst);
    // cv::imshow("edge", dst);

}

std::pair<Indices, Indices> Hough_lane_detection::divideLines(const Lines& lines)
{
    Indices leftLineIndices;
    Indices rightLineIndices;

    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    float slope;

    cv::Vec4f l;
    for (size_t i = 0; i < lines.size(); i++) {
        l = lines[i];
        
        // 0 1 2 3 x1 y1 x2 y2
        x1 = l[0]; 
        y1 = l[1];
        // point of start(top)
        x2 = l[2]; 
        y2 = l[3];
        // point of end(bottom)

        if(y1 > y2){
            std::swap(y1, y2);
            std::swap(x1, x2);
        }


        if (y2 < mROIHeight - mMargin) continue;

        if(x1 - x2 == 0 || y1 - y2 == 0){
            slope = 0;
        }else{
            slope = static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
        }
        
        if(slope == 0 || abs(slope) < 0.2) continue; 
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


std::pair<int32_t, int32_t> Hough_lane_detection::getLanePosition(){
    static constexpr int32_t hough_rho = 1;
    static constexpr int32_t hough_min = 20;
    static constexpr int32_t hough_max_gab = 0;

    int32_t leftPositionX = 0;
    int32_t rightPositionX = 0;

    // hough
    cv::HoughLinesP(mFrame_roi, mLines, hough_rho, CV_PI / 180, hough_min, hough_max_gab, mROIHeight-mMargin);

    for(Line l : mLines){
        // 0 1 2 3 x1 y1 x2 y2
        int32_t x1 = l[0]; 
        int32_t y1 = l[1];
        // point of start(top)
        int32_t x2 = l[2]; 
        int32_t y2 = l[3];
        // point of end(bottom)
    }



    mDivideLines = divideLines(mLines);
    // leftPositionX = getLinePositionX(lines, mDivideLines.first, Direction::LEFT);
    // rightPositionX = getLinePositionX(lines, mDivideLines.second, Direction::RIGHT);

	// if (mDebugging)
	// {
    //     frame.copyTo(mDebugFrame);
	// 	drawLines(lines, divide_lines.first, divide_lines.second);
	// }

    return { leftPositionX, rightPositionX };
}

void Hough_lane_detection::run(){

    mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);

    while (true)
    {   

        mCap >> mFrame;
        // std::cout << mCap.get(cv::CAP_PROP_POS_FRAMES) << "\n";
        preProcessing();
        // getLanePosition();

        if (mFrame.empty())
            continue;

        // cv::imshow("frame", mFrame);
        // cv::imshow("roi", mFrame_roi);
        
        if(cv::waitKey(1) == 27) break;
        if(mCap.get(cv::CAP_PROP_POS_FRAMES) == 1800){
            mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);
        }
    }

    mCap.release();
    cv::destroyAllWindows();
}


