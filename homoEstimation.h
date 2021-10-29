/*******************************************************************************
 *
 * \file    homoEstimation.h
 * \brief   单应性估计模块
 * \author  1851738杨皓冬
 * \version 2.0
 * \date    2021-06-11
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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "publicElement.h"
#include"ransac_personal.h"
#include <iostream>
using namespace cv;
using namespace std;

#pragma once
#ifndef HOMOESTIMATION_H
#define HOMOESTIMATION_H

class homoEst
{
public:
    typedef struct
    {
        Point2f left_top;
        Point2f left_bottom;
        Point2f right_top;
        Point2f right_bottom;
    }homo_corners;

    homo_corners corners;                       // 单应性变换后图像的四个角
    vector<Point2f> srcPoints_1, srcPoints_2;   // 映射点集
    int imgHeight;                              // 图像高 .pix
    int imgWidth;                               // 图像宽 .pix
    Mat H;                                      // 单应性矩阵
    int rightBound;                             // 单应变换后图像的右边界
    int leftBound;                              // 单应变换后图像的左边界
    int topBound;                               // 单应变换后图像的上边界
    int bottomBound;                            // 单应变换后图像的下边界

public:
    /*
     * @breif:构造函数
     * @prama[in]:InputArray srcPoints_1, InputArray srcPoints_2->输入映射点集(至少4对)
     * @prama[in]:MatSize imgSize->源图像尺寸
     */
    homoEst(vector<Point2f> srcPoints_1, vector<Point2f> srcPoints_2, MatSize imgSize);
    homoEst();

    /*
     * @breif:打印映射后图像的四角点坐标、打印变换后图像边界像素
     * @prama[in]:None
     * @retval:None
     */
    void printCorner();
    void printBound();

    /*
     * @breif:根据映射点对，求源图像间的单应性矩阵(基本)
     * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
     * @retval:None
     */
    void findHomography_Base(int dir=1);

    /*
     * @breif:计算单应性变换后图像的边界像素
     * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
     * @retval:None
     */
    void calTransBound(int dir=1);

    /*
     * @breif:获取经过单应变换后的图像
     * @prama[in]:srcImg->变换前的原图像；H->单应变换矩阵; mapSize->变换后图像的大小；debug->调试模式
     * @retval:dstImg->变换后的图像
     */
    Mat imgMapByHomo(Mat& srcImg, Mat& H, Size mapSize, int debug= DEBUGMODE_NORMAL);

private:
    /*
     * @breif:计算单应性变换后图像的四个角坐标
     * @prama[in]:dir:1->从src1到src2的映射(默认),dir:0->从src2到src1的映射
     * @retval:None
     */
    void calCorners(int dir = 1);
};

#endif // !HOMOESTIMATION_H
