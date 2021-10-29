/*******************************************************************************
 *
 * \file    featureDesc.cpp
 * \brief   图像特征描述
 * \author  1851738杨皓冬
 * \version 2.0
 * \date    2021-06-17
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-11  | v1.0    | 1851738杨皓冬  |
 * 2021-06-11  | v2.0    | 1853735赵祉淇  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include "featureDesc.h"

/*
 * @breif:特征描述，基于ORB算法
 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
 * @retval:None
 */
void featureDesc::getFeatureDesc_ORB(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc)
{
	Ptr<ORB> OrbFeature = ORB::create();
	OrbFeature->detectAndCompute(srcGray, Mat(), keyPoint, Desc);
}

/*
 * @breif:特征描述，基于SIFT算法
 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
 * @retval:None
 */
void featureDesc::getFeatureDesc_SIFT(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc)
{
	Ptr<Feature2D> siftFeature = SIFT::create();
	siftFeature->detect(srcGray, keyPoint);
	siftFeature->compute(srcGray, keyPoint, Desc);
}

/*
 * @breif:特征描述，基于SURF算法
 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
 * @retval:None
 */
//void featureDesc::getFeatureDesc_SURF(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc)
//{
//	Ptr<cv::xfeatures2d::SURF> surfFeature = cv::xfeatures2d::SURF::create(1000);
//	surfFeature->detect(srcGray, keyPoint);
//	surfFeature->compute(srcGray, keyPoint, Desc);
//}

/*
 * @breif:特征描述，基于BRISK算法
 * @prama[in]:srcGray->源图像的灰度图; keyPoint->输出检测的特征点; Desc->输出特征点对应的描述子
 * @retval:None
 */
void featureDesc::getFeatureDesc_BRISK(Mat& srcGray, vector<KeyPoint>& keyPoint, Mat& Desc)
{
	Ptr<Feature2D> BriskFeature = BRISK::create();
	BriskFeature->detect(srcGray, keyPoint);
	BriskFeature->compute(srcGray, keyPoint, Desc);
}