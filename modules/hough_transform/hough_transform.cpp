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
    
    mYAxisMargin = 10;

}

cv::Mat Hough_lane_detection::preProcessing(cv::Mat src)
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
    src=canny(src);

    return src;
}


std::pair<Indices, Indices> Hough_lane_detection::divideLines(const Lines& lines)
{
    Indices leftLineIndices;
    Indices rightLineIndices;

    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    float32_t slope;

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

        // 시작점 필터링
        if (y2 < mROIHeight - mYAxisMargin) continue;

        if(x1 - x2 == 0 || y1 - y2 == 0){
            slope = 0;
        }else{
            slope = static_cast<float32_t>(y2 - y1) / static_cast<float32_t>(x2 - x1);
        } 

        // if(slope == 0 || abs(slope) < 0.2) continue; 
        if ((fabsf(slope) < std::numeric_limits<float32_t>::epsilon()) || abs(slope) < 0.2) continue; 

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


int32_t Hough_lane_detection::getLinePositionX(const Lines& lines, const Indices& lineIndices, bool direction)
{
    int32_t positionX = 0;

    auto [m, b] = getLineParameters(lines, lineIndices);

    if((fabsf(m) < std::numeric_limits<float>::epsilon()) && (fabsf(b) < std::numeric_limits<float>::epsilon()))
    {
        positionX = (direction == true) ? 0 : mImageWidth;
    }
    else
    {
        int32_t y = static_cast<int32_t>(mROIHeight / 2);
        positionX = static_cast<int32_t>((y - b) / m);
        // b += mROIStartHeight;
    }

    return positionX;
}


std::pair<float, float> Hough_lane_detection::getLineParameters(const Lines& lines, const Indices& lineIndices)
{
    float m = 0.0F;
    float b = 0.0F;

    float x_sum = .0F;
    float y_sum = .0F;
    float m_sum = .0F;
    Line l;

    int32_t size = lineIndices.size();
    if (size == 0)
    {
        return std::pair<float, float>(0, 0);
    }

    float x1, y1, x2, y2;
    for (int32_t i : lineIndices) {
        l = lines[i];
        x1 = l[0];
        y1 = l[1];
        x2 = l[2];
        y2 = l[3];

        x_sum += x1 + x2;
        y_sum += y1 + y2;
        m_sum += static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
    }

    float x_avg = x_sum / (size * 2);
    float y_avg = y_sum / (size * 2);
    m = m_sum / size;
    b = y_avg - m * x_avg;

    return { m, b };
}



void Hough_lane_detection::getTrackPointX(int32_t positionX, int32_t& trackPosX, float areaMin, float areaMax){
    // if(positionX > mImageWidth / 2) 
    //     trackPosX = positionX-50;
    //     return ;

    // if(positionX < 0) 
    //     trackPosX = 50;
    //     return;

    // if((trackPosX - 10 >= positionX) && (trackPosX + 10 <= positionX)){
    //     trackPosX = positionX;
    //     return;
    // }
    // if((mImageWidth*area-50 <= positionX) && (positionX <= mImageWidth*area+50)){
    //     trackPosX = positionX;
    //     return ;
    // }


    if((areaMin <= positionX) && (positionX <= areaMax)){
        trackPosX = positionX;
        return ;
    }


    // if(! (positionX < mImageWidth /2)) return;


    
    // else if((leftTrackPosx - 10 <= positionX) && (leftTrackPosx + 10 <= positionX)){
    //     leftTrackPosx = positionX;
    //     return;
    // }

    
}



void Hough_lane_detection::run(){
    static constexpr int32_t hough_rho = 1;
    static constexpr int32_t hough_min = 20;
    static constexpr int32_t hough_max_gab = 0;

    static int32_t left_track_start_point = 10;
    static int32_t left_track_end_point = mImageWidth*0.25+50;
    static int32_t right_track_start_point = mImageWidth*0.75-50;
    static int32_t right_track_end_point = mImageWidth-10;


    cv::Mat frame, frame_roi;
    Lines lines;
    int32_t leftPositionX = 0;
    int32_t rightPositionX = 0;

    int32_t leftTrackPosx = 0;
    int32_t rightTrackPosx = 0;
    
    // mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);

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

        auto [hLeftLineIndices, hRightLineIndices] = divideLines(lines);
        leftPositionX = getLinePositionX(lines, hLeftLineIndices, true);
        rightPositionX = getLinePositionX(lines, hRightLineIndices, false);

        getTrackPointX(leftPositionX, leftTrackPosx, left_track_start_point, left_track_end_point);
        getTrackPointX(rightPositionX, rightTrackPosx, right_track_start_point, right_track_end_point);


        // cv::Rect ltrack_box(cv::Point(leftPositionX,mROIStartHeight+mROIHeight), cv::Point(leftPositionX,mROIStartHeight+mROIHeight+10));

        // cv::Rect ltrack_box(cv::Point(mImageWidth*0.25-50,mROIStartHeight+mROIHeight), cv::Point(mImageWidth*0.25+50,mROIStartHeight+mROIHeight+10));
        // cv::Rect rtrack_box(cv::Point(mImageWidth*0.75-50,mROIStartHeight+mROIHeight), cv::Point(mImageWidth*0.75+50,mROIStartHeight+mROIHeight+10));

        for(auto i : hLeftLineIndices){
            Line l  = lines[i];
            int32_t x1 = l[0];
            int32_t y1 = l[1];
            int32_t x2 = l[2];
            int32_t y2 = l[3];
            cv::line(frame, cv::Point(x1,y1+mROIStartHeight), cv::Point(x2,y2+mROIStartHeight), cv::Scalar(255,0,0), 3, cv::LINE_AA);
        }

        for(auto i : hRightLineIndices){
            Line l  = lines[i];
            int32_t x1 = l[0];
            int32_t y1 = l[1];
            int32_t x2 = l[2];
            int32_t y2 = l[3];
            cv::line(frame, cv::Point(x1,y1+mROIStartHeight), cv::Point(x2,y2+mROIStartHeight), cv::Scalar(255,0,0), 3, cv::LINE_AA);
        }

        cv::rectangle(frame, cv::Point(left_track_start_point,0), cv::Point(left_track_end_point,480), cv::Scalar(255,0,0), 3, cv::LINE_AA);
        cv::rectangle(frame, cv::Point(right_track_start_point,0), cv::Point(right_track_end_point,480), cv::Scalar(255,0,0), 3, cv::LINE_AA);
        

        cv::circle(frame, cv::Point(leftPositionX, mROIStartHeight+mROIHeight), 3, cv::Scalar(255,255,255), 3, cv::LINE_AA);
        cv::circle(frame, cv::Point(rightPositionX, mROIStartHeight+mROIHeight), 3, cv::Scalar(255,255,255), 3, cv::LINE_AA);
        // cv::circle(frame, cv::Point(leftPositionX, mROIHeight / 2), 3, cv::Scalar(255,255,255), 3, cv::LINE_AA);
        // cv::circle(frame, cv::Point(rightPositionX, mROIHeight / 2), 3, cv::Scalar(255,255,255), 3, cv::LINE_AA);

        cv::circle(frame, cv::Point(leftTrackPosx, mROIStartHeight+mROIHeight), 3, cv::Scalar(0,0,255), 3, cv::LINE_AA);
        cv::circle(frame, cv::Point(rightTrackPosx, mROIStartHeight+mROIHeight), 3, cv::Scalar(0,255,0), 3, cv::LINE_AA);
        cv::circle(frame, cv::Point((leftTrackPosx+rightTrackPosx)/2, mROIStartHeight+mROIHeight), 3, cv::Scalar(255,255,0), 3, cv::LINE_AA);
        // cv::circle(frame, cv::Point((leftPositionX+rightPositionX)/2, mROIStartHeight+mROIHeight), 3, cv::Scalar(0,255,255), 3, cv::LINE_AA);

        // cv::rectangle(frame, ltrack_box, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        // cv::rectangle(frame, rtrack_box, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        // leftPositionX = getLinePositionX(lines, hLeftLineIndices, Direction::LEFT);
        // rightPositionX = getLinePositionX(lines, hRightLineIndices, Direction::RIGHT);

        cv::imshow("frame", frame);
        cv::imshow("roi", frame_roi);
        
        if(cv::waitKey(20) == 27) break;
        if(mCap.get(cv::CAP_PROP_POS_FRAMES) == 1800){
            mCap.set(cv::CAP_PROP_POS_FRAMES, 1500);
        }
    }

    mCap.release();
    cv::destroyAllWindows();
}


