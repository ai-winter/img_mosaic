/*******************************************************************************
 *
 * \file    featureDesc.h
 * \brief   图像特征描述
 * \author  1851738杨皓冬  +   1853735赵祉淇
 * \version 3.0
 * \date    2021-06-17
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-11  | v2.0    | 1851738杨皓冬  |
 * 2021-06-17  | v3.0    | 1853735赵祉淇  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace std;

#pragma once
#ifndef FEATUREDESC_H
#define FEATUREDESC_H

class featureDesc
{
public:
	/*
	 * @breif:特征描述，基于ORB算法
	 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
	 * @retval:None
	 */
	void getFeatureDesc_ORB(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc);

	/*
	 * @breif:特征描述，基于SIFT算法
	 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
	 * @retval:None
	 */
	void getFeatureDesc_SIFT(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc);

	/*
	 * @breif:特征描述，基于BRISK算法
	 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
	 * @retval:None
	 */
	void getFeatureDesc_BRISK(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc);
	/*
	 * @breif:特征描述，基于SURF算法
	 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
	 * @retval:None
	 */
	void getFeatureDesc_SURF(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc);
};


#endif // !FEATUREDESC_H