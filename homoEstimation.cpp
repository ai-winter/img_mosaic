/*******************************************************************************
 *
 * \file    homoEstimation.cpp
 * \brief   单应性估计模块
 * \author  1851738杨皓冬  +   1853735赵祉淇
 * \version 3.0
 * \date    2021-06-17
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-09  | v1.0    | 1851738杨皓冬  |
 * 2021-06-11  | v2.0    | 1851738杨皓冬  |
 * 2021-06-17  | v3.0    | 1853735赵祉淇  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include "homoEstimation.h"

/*===================================================================================*/
/******************************* 公有函数 *********************************************/
/*===================================================================================*/

 /*
  * @breif:构造函数
  * @prama[in]:InputArray srcPoints_1, InputArray srcPoints_2->输入映射点集(至少4对)
  * @prama[in]:MatSize imgSize->源图像尺寸
  */
homoEst::homoEst(vector<Point2f> srcPoints_1, vector<Point2f> srcPoints_2, MatSize imgSize)
{
	homoEst::srcPoints_1 = srcPoints_1;
	homoEst::srcPoints_2 = srcPoints_2;
	homoEst::imgHeight = imgSize[0];
	homoEst::imgWidth = imgSize[1];
}

/*
 * @breif:打印映射后图像的四角点坐标、打印变换后图像边界像素
 * @prama[in]:None
 * @retval:None
 */
void homoEst::printCorner()
{
	cout << "左上角:" << homoEst::corners.left_top << endl;
	cout << "左下角:" << homoEst::corners.left_bottom << endl;
	cout << "右上角:" << homoEst::corners.right_top << endl;
	cout << "右下角:" << homoEst::corners.right_bottom << endl;
}
void homoEst::printBound()
{
    cout << "左边界:" << homoEst::leftBound << endl;
    cout << "右边界:" << homoEst::rightBound << endl;
    cout << "上边界:" << homoEst::topBound << endl;
    cout << "下边界:" << homoEst::bottomBound << endl;
}

cv::Mat find_H_matrix(std::vector<cv::Point2f> src, std::vector<cv::Point2f> tgt) {

    std::cout << "calculating H matrix..." << std::endl;

    double Point_matrix[8][8];
    for (int i = 0; i < 4; i++) {
        double row1[8] = { src[i].x, src[i].y, 1,        0,        0, 0, -tgt[i].x * src[i].x, -tgt[i].x * src[i].y };
        double row2[8] = { 0       ,        0, 0, src[i].x, src[i].y, 1, -tgt[i].y * src[i].x, -tgt[i].y * src[i].y };

        /* save 4 point data to 8*8 Matrix */
        memcpy(Point_matrix[2 * i], row1, sizeof(row1));
        memcpy(Point_matrix[2 * i + 1], row2, sizeof(row2));
    }

    /* 8*8 Matrix */
    cv::Mat Point_data(8, 8, CV_64FC1, Point_matrix);
    cv::Mat Point_target = (cv::Mat_<double>(8, 1) << tgt[0].x, tgt[0].y, tgt[1].x, tgt[1].y, tgt[2].x, tgt[2].y, tgt[3].x, tgt[3].y);
    cv::Mat h8 = Point_data.inv() * Point_target;

    /* 3*3 H Matrix */
    double H8[9];
    for (int i = 0; i < 8; i++) {
        H8[i] = h8.at<double>(i, 0);
    }
    H8[8] = 1.0;
    cv::Mat H(3, 3, CV_64FC1, H8);

    return H.clone();
}

cv::Mat find_H_SVD(std::vector<cv::Point2f> src, std::vector<cv::Point2f> tgt) {

    double Point_matrix[8][9];
    for (int i = 0; i < 4; i++) {
        double row1[9] = { src[i].x, src[i].y, 1,        0,        0, 0, -tgt[i].x * src[i].x, -tgt[i].x * src[i].y, -tgt[i].x };
        double row2[9] = { 0       ,        0, 0, src[i].x, src[i].y, 1, -tgt[i].y * src[i].x, -tgt[i].y * src[i].y, -tgt[i].y };

        /* save 4 point data to 8*8 Matrix */
        memcpy(Point_matrix[2 * i], row1, sizeof(row1));
        memcpy(Point_matrix[2 * i + 1], row2, sizeof(row2));
    }

    cv::Mat Point_data(8, 9, CV_64FC1, Point_matrix);
    cv::SVD svd(Point_data, cv::SVD::FULL_UV);
    cv::Mat V = svd.vt.t();

    double h[9] = { 0 };
    for (int i = 0; i < V.rows; i++) {
        h[i] = V.at<double>(i, V.cols - 1);
        h[i] = h[i] / V.at<double>(V.rows - 1, V.cols - 1);    /* Normalized */
    }

    cv::Mat H(3, 3, CV_64FC1, h);

    return H.clone();
}


/*
 * @breif:根据映射点对，求源图像间的单应性矩阵(基本)
 * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
 * @retval:None
 */
//已经更改为自定义的RANSAC计算对应矩阵部分
void homoEst::findHomography_Base(int dir)
{
    //适用于RANSAC算法的变量定义
    Mat H_32;
    vector<size_t> best_inliers;
    //使用自定义RANSAC方法计算
    if (dir) GetHomographyRANSAC(homoEst::srcPoints_1, homoEst::srcPoints_2,4,H_32,best_inliers,3,2000, 0.995);
    else GetHomographyRANSAC(homoEst::srcPoints_2, homoEst::srcPoints_1, 4, H_32, best_inliers, 3, 2000, 0.995);
    H_32.convertTo(homoEst::H,CV_64F,1,0);
    
    //使用线性化方法计算
    /*if (dir)	homoEst::H = find_H_matrix(homoEst::srcPoints_1, homoEst::srcPoints_2);
    else	homoEst::H = find_H_matrix(homoEst::srcPoints_2, homoEst::srcPoints_1);*/

    //使用SVD计算
    /*if(dir)	homoEst::H = find_H_SVD(homoEst::srcPoints_1, homoEst::srcPoints_2);
    else	homoEst::H = find_H_SVD(homoEst::srcPoints_2, homoEst::srcPoints_1);*/

    //直接调用库函数
	/*if(dir)	homoEst::H = findHomography(homoEst::srcPoints_1, homoEst::srcPoints_2);
	else	homoEst::H = findHomography(homoEst::srcPoints_2, homoEst::srcPoints_1);*/
}

/*
 * @breif:计算单应性变换后图像的边界像素
 * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
 * @retval:None
 */
void homoEst::calTransBound(int dir)
{
    homoEst::calCorners(dir);
    homoEst::leftBound = cmpMin(homoEst::corners.left_top.x, homoEst::corners.left_bottom.x);
    homoEst::rightBound = cmpMax(homoEst::corners.right_top.x, homoEst::corners.right_bottom.x);
    homoEst::bottomBound = cmpMin(homoEst::corners.left_bottom.y, homoEst::corners.right_bottom.y);
    homoEst::topBound = cmpMax(homoEst::corners.left_top.y, homoEst::corners.right_top.y);
}

/*
 * @breif:获取经过单应变换后的图像
 * @prama[in]:srcImg->变换前的原图像；H->单应变换矩阵; mapSize->变换后图像的大小；debug->调试模式
 * @retval:dstImg->变换后的图像
 */
Mat homoEst::imgMapByHomo(Mat& srcImg, Mat& H, Size mapSize, int debug)
{
    Mat dstImg;
    warpPerspective(srcImg, dstImg, H, mapSize);
    if (debug)      imshow("homoEst::imgMapByHomo", dstImg);
    return dstImg;
}
/*-----------------------------------------------------------------------------------*/


/*===================================================================================*/
/******************************* 私有函数 *********************************************/
/*===================================================================================*/

/*
 * @breif:计算单应性变换后图像的四个角坐标
 * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
 * @retval:None
 */
void homoEst::calCorners(int dir)
{
    //左上、左下、右上、右下
    Mat srcCorner = (Mat_<double>(3, 4) << 0, 0, homoEst::imgWidth, homoEst::imgWidth,
        0, homoEst::imgHeight, 0, homoEst::imgHeight,
        1, 1, 1, 1);
    Mat dstCorner = homoEst::H * srcCorner;
    homoEst::corners.left_top.x = dstCorner.at<double>(0, 0) / dstCorner.at<double>(2, 0);
    homoEst::corners.left_top.y = dstCorner.at<double>(1, 0) / dstCorner.at<double>(2, 0);
    homoEst::corners.left_bottom.x = dstCorner.at<double>(0, 1) / dstCorner.at<double>(2, 1);
    homoEst::corners.left_bottom.y = dstCorner.at<double>(1, 1) / dstCorner.at<double>(2, 1);
    homoEst::corners.right_top.x = dstCorner.at<double>(0, 2) / dstCorner.at<double>(2, 2);
    homoEst::corners.right_top.y = dstCorner.at<double>(1, 2) / dstCorner.at<double>(2, 2);
    homoEst::corners.right_bottom.x = dstCorner.at<double>(0, 3) / dstCorner.at<double>(2, 3);
    homoEst::corners.right_bottom.y = dstCorner.at<double>(1, 3) / dstCorner.at<double>(2, 3);
}

/*-----------------------------------------------------------------------------------*/