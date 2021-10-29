/*******************************************************************************
 *
 * \file    imgProcess.h
 * \brief   图像处理与美化：读入、裁剪、灰度化、拼接等
 * \author  1851738杨皓冬
 * \version 3.0
 * \date    2021-06-12
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-09  | v1.0    | 1851738杨皓冬  |
 * 2021-06-11  | v2.0    | 1851738杨皓冬  |
 * 2021-06-12  | v3.0    | 1851738杨皓冬  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "publicElement.h"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

/*===================================================================================*/
/******************************** 宏定义 *********************************************/
/*===================================================================================*/
#define SHOWMODE_GRAY			   0							// 灰度模式
#define SHOWMODE_RGB			   1							// 彩色模式
#define PYRWIDTH				  736							// 图像金字塔原图片宽度
#define PYRHEIGHT				  240							// 
/*-----------------------------------------------------------------------------------*/

#pragma once
#ifndef IMGPROCESS_H
#define IMGPROCESS_H

class imgProcess
{
public:
	vector<Mat> RGBImgs;							// 图片集合(RGB)
	vector<Mat> GrayImgs;							// 图片集合(灰度)
	int imgNum;										// 图片数量

public:
	/*
	 * @breif:构造函数,构造路径下的彩色与对应灰度图片集
	 * @prama[in]:string srcFileTxt->.txt格式的源图片路径文件
	 */
	imgProcess();
	imgProcess(string srcFileTxt);

	/*
	 * @breif:原图显示
	 * @prama[in]:mode->
	 * @retval:dstImg->拼接后的图像
	 */
	void showSrcImg(int mode = SHOWMODE_RGB);

	/*
	 * @breif:图像拼接
	 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; debug->调试模式
	 * @retval:dstImg->拼接后的图像
	 */
	Mat imgMosaic(Mat& leftImg, Mat& rightImg, int debug = DEBUGMODE_NORMAL);

	/*
	 * @breif:将图像规范到某个大小，超出原图的部分用黑色像素填充
	 * @prama[in]:srcImg->原图像; height,width->规范的宽高;
	 * @retval:dstImg->规范后的图像
	 */
	Mat imgCanonical(const Mat srcImg, int height, int width);

	/*
	 * @breif:图像γ调整
	 * @prama[in]:srcImg->原图像; gamma->γ值;
	 * @note:γ公式->O=(I/255)^γ ×255
	 * @retval:dstImg->调整后的图像
	 */
	Mat imgGammaProcess(Mat& srcImg, double gamma);

	/*
	 * @breif:拼接处优化，采用alpha优化方法
	 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; dstImg->拼接后图像——优化对象; 
	 * @prama[in]:start->优化区域起点;end->优化区域终点;debug->调试模式
	 * @retval:None
	 */
	void seamOpt_alpha(Mat& leftImg, Mat& rightImg,Mat& dstImg, int start, int end, int debug = DEBUGMODE_NORMAL);

	/*
	 * @breif:拼接处优化，采用Laplace优化方法
	 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; dstImg->拼接后图像——优化对象;
	 * @prama[in]:threshold->优化阈值;debug->调试模式
	 * @retval:None
	 */
	void seamOpt_laplace(Mat leftImg, Mat rightImg, Mat& dstImg, float threshold, int debug);

private:
	/*
	 * @breif:建立高斯金字塔
	 * @prama[in]:srcImg->待高斯金字塔化的源图像;imgPyr->金字塔化的图像集合;level->金字塔层数
	 * @retval:None
	 */
	void buildGaussPyr(Mat srcImg, vector<Mat>& imgPyr, int level);

	/*
	 * @breif:建立拉普拉斯金字塔
	 * @prama[in]:imgGaussPyr->图像的高斯金字塔;imgLaplacePyr->输出的图像拉普拉斯金字塔;level->金字塔层数
	 * @retval:None
	 */
	void buildLaplacePyr(const vector<Mat> imgGaussPyr, vector<Mat>& imgLaplacePyr, int level);

	/*
	 * @breif:建立融合拉普拉斯金字塔
	 * @prama[in]:imgLp_1、imgLp_2->待融合图像的拉普拉斯金字塔;maskGauss->掩码的高斯金字塔
	 * @prama[in]:blendLp->输出的融合拉普拉斯金字塔
	 * @retval:None
	 */
	void blendLaplacePyr(const vector<Mat> imgLp_1, const vector<Mat> imgLp_2, const vector<Mat> maskGauss,
		vector<Mat>& blendLp);

	/*
	 * @breif:图像拉普拉斯融合
	 * @prama[in]:imgHighest->图像混合的起点,即两个待融合图像高斯金字塔最高层按mask加权求和的结果
	 * @prama[in]:blendLp->输出的融合拉普拉斯金字塔
	 * @retval:dstImg->融合的图像
	 */
	Mat imgLaplaceBlend(Mat& imgHighest, vector<Mat> blendLp);
};

#endif // !PREPROCESS_H
