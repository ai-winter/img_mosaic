/*******************************************************************************
 *
 * \file    featureMatch.h
 * \brief   图像特征匹配
 * \author  1851738杨皓冬
 * \version 1.0
 * \date    2021-06-11
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-11  | v2.0    | 1851738杨皓冬  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;
using namespace cvflann;
using namespace flann;

/*===================================================================================*/
/******************************** 宏定义 *********************************************/
/*===================================================================================*/
#define MAXNUMBER		   10000					// 大数
#define MATCHMODE_HAMMING  0						// 汉明距离匹配模式
#define MATCHMODE_NORML2   1						// 二次范数匹配模式
/*-----------------------------------------------------------------------------------*/

#pragma once
#ifndef FEATUREMATCH_H
#define FEATUREMATCH_H

class featureMatch
{
public:
	/*
	 * @breif:特征匹配，基于Low's算法
	 * @prama[in]:Desc_1,Desc_2->待匹配图片的特征描述子,threshold->阈值,matchMode->匹配模式,宏定义
	 * @retval:GoodMatchPoints->筛选出的优秀特征点匹配对
	 */
	vector<DMatch> featureMatch_Lows(const Mat Desc_1, const Mat Desc_2, float threshold, int matchMode);

	/*
	 * @breif:特征匹配，基于minMax算法
	 * @prama[in]:Desc_1,Desc_2->待匹配图片的特征描述子,threshold->阈值,matchMode->匹配模式,宏定义
	 * @retval:GoodMatchPoints->筛选出的优秀特征点匹配对
	 */
	vector<DMatch> featureMatch_MinMax(const Mat Desc_1, const Mat Desc_2, float threshold, int matchMode);

	//void drawMatchImg();

	/*
	 * @breif:获得优秀特征点对对应的原图像素坐标
	 * @prama[in]:goodMatchPoints->筛选出的优秀特征点匹配对;keyPtLeft,keyPtRight->左右特征点集
	 * @prama[in]:goodPtLeft,goodPtRight->左右优秀特征点
	 * @retval:None
	 */
	void getGoodPt(vector<DMatch> goodMatchPoints, vector<KeyPoint> keyPtRight, vector<KeyPoint>keyPtLeft,
		vector<Point2f>&goodPtRight, vector<Point2f>&goodPtLeft);

private:
	/*
	 * @breif:匹配模式转换为Flann、BFM
	 * @prama[in]:matchMode->int格式匹配模式
	 * @retval:matchMode->flann_distance_t或int格式匹配模式
	 */
	flann_distance_t matchModeTransFlann(int matchMode);
	int matchModeTransBFM(int matchMode);

	/*
	 * @breif:获得二者中较小、较大的描述子
	 * @prama[in]:Desc_1、Desc_2->特征描述子
	 * @retval:smallDesc or largeDesc
	 */
	Mat getSmallDesc(const Mat Desc_1, const Mat Desc_2);
	Mat getLargeDesc(const Mat Desc_1, const Mat Desc_2);
};

#endif // !FEATUREMATCH_H