/*******************************************************************************
 *
 * \file    imgProcess.cpp
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
#include "imgProcess.h"

 /*===================================================================================*/
 /******************************* 公有函数 *********************************************/
 /*===================================================================================*/

 /*
  * @breif:构造函数,构造路径下的彩色与对应灰度图片集
  * @prama[in]:string srcFileTxt->.txt格式的源图片路径文件
  */
imgProcess::imgProcess(string srcFileTxt)
{
	ifstream file(srcFileTxt);
	string img_name;
	while (getline(file, img_name))
	{
		Mat tempRGBImg = imread(img_name, 1);
		imgProcess::RGBImgs.push_back(tempRGBImg);
		Mat tempGrayImg;
		cvtColor(tempRGBImg, tempGrayImg, COLOR_RGB2GRAY);
		imgProcess::GrayImgs.push_back(tempGrayImg);
	}
	imgProcess::imgNum = imgProcess::RGBImgs.size();
	imgProcess::RGBImgs[0] = imgProcess::imgGammaProcess(imgProcess::RGBImgs[0], 0.9);
}

/*
 * @breif:原图显示
 * @prama[in]:mode->
 * @retval:dstImg->拼接后的图像
 */
void imgProcess::showSrcImg(int mode)
{
	if (mode)
	{
		for (int i = 0; i < imgProcess::imgNum; i++)
		{
			string tempWinName = getFormatStr("彩色图片%d", i+1);
			imshow(tempWinName, imgProcess::RGBImgs[i]);
		}
	}
	else
	{
		for (int i = 0; i < imgProcess::imgNum; i++)
		{
			string tempWinName = getFormatStr("灰度图片%d", i + 1);
			imshow(tempWinName, imgProcess::GrayImgs[i]);
		}
	}
}

/*
 * @breif:图像拼接
 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; debug->调试模式
 * @retval:dstImg->拼接后的图像
 */
Mat imgProcess::imgMosaic(Mat& leftImg, Mat& rightImg, int debug)
{
	//创建拼接后的图,需提前计算图的大小
	int dstWidth = cmpMax(leftImg.cols, rightImg.cols);		// 取最宽长度为拼接图的宽度
	int dstHeight = cmpMax(leftImg.rows, rightImg.rows);		// 取最高长度为拼接图的长度

	Mat dstImg(dstHeight, dstWidth, CV_8UC3);
	dstImg.setTo(0);
	rightImg.copyTo(dstImg(Rect(0, 0, rightImg.cols, rightImg.rows)));
	leftImg.copyTo(dstImg(Rect(0, 0, leftImg.cols, leftImg.rows)));
	if (debug)			imshow("imgProcess::imgMosaic", dstImg);
	return dstImg;
}

/*
 * @breif:将图像规范到某个大小，超出原图的部分用黑色像素填充
 * @prama[in]:srcImg->原图像; height,width->规范的宽高;
 * @retval:dstImg->规范后的图像
 */
Mat imgProcess::imgCanonical(const Mat srcImg, int height, int width)
{
	Mat dstImg(height, width, CV_8UC3, Scalar(0, 0, 0));			//创建一个全黑的图片
	Mat imageROI = dstImg(Rect(0, 0, srcImg.cols, srcImg.rows));
	Mat mask;
	cvtColor(srcImg, mask, COLOR_RGB2GRAY);
	srcImg.copyTo(imageROI, mask);									//将原图叠加到ROI
	return dstImg;
}

/*
 * @breif:图像γ调整
 * @prama[in]:srcImg->原图像; gamma->γ值;
 * @note:γ公式->O=(I/255)^γ ×255
 * @retval:dstImg->调整后的图像
 */
Mat imgProcess::imgGammaProcess(Mat& srcImg, double gamma)
{
	Mat dstImg;
	Mat lookupTable(1, 256, CV_8U);
	uchar* p = lookupTable.ptr();
	for (int i = 0; i < 256; i++) {
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	}
	LUT(srcImg, lookupTable, dstImg);
	return dstImg;
}

/*
 * @breif:拼接处优化，采用alpha优化方法
 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; dstImg->拼接后图像——优化对象;
 * @prama[in]:start->优化区域起点;end->优化区域终点;debug->调试模式
 * @retval:None
 */
void imgProcess::seamOpt_alpha(Mat& leftImg, Mat& rightImg, Mat& dstImg, int start, int end,int debug)
{
    double processWidth = end - start;              // 优化处理区域，即重叠区域宽度  
    double alpha = 1;								// 左图leftImg中像素的权重
    for (int i = 0; i < dstImg.rows; i++)
    {
        uchar* rowAddrLeft = leftImg.ptr<uchar>(i);	// 获取图像第i行的首地址
        uchar* rowAddrRight = rightImg.ptr<uchar>(i);
        uchar* rowAddrDst = dstImg.ptr<uchar>(i);
        for (int j = start; j < leftImg.cols; j++)
        {
            // 如果遇到右图中无像素的黑点，则完全拷贝左图像素
            if (rowAddrRight[j * 3] == 0 && rowAddrRight[j * 3 + 1] == 0 && rowAddrRight[j * 3 + 2] == 0)  alpha = 1;
			// 左图中像素的权重，与当前处理点距重叠区域左边界的距离成正比
            else	alpha = (processWidth - (j - start)) / processWidth; 
			rowAddrDst[j * 3] = rowAddrLeft[j * 3] * alpha + rowAddrRight[j * 3] * (1 - alpha);
			rowAddrDst[j * 3 + 1] = rowAddrLeft[j * 3 + 1] * alpha + rowAddrRight[j * 3 + 1] * (1 - alpha);
			rowAddrDst[j * 3 + 2] = rowAddrLeft[j * 3 + 2] * alpha + rowAddrRight[j * 3 + 2] * (1 - alpha);
        }
		if (debug)		imshow("imgProcess::seamOpt_alpha", dstImg);
    }
}

/*
 * @breif:拼接处优化，采用Laplace优化方法
 * @prama[in]:leftImg->左拼接图像; rightImg->右拼接图像; dstImg->拼接后图像——优化对象;
 * @prama[in]:threshold->优化阈值;debug->调试模式
 * @retval:None
 */
void imgProcess::seamOpt_laplace(Mat leftImg, Mat rightImg, Mat& dstImg, float threshold, int debug)
{
	vector<Mat> gaussPyrLeft, gaussPyrRight, laplacePyrLeft, laplacePyrRight;		// 声明高斯金字塔数据结构
	vector<Mat> maskGaussPyr;														// 声明掩码的高斯金字塔
	vector<Mat> blendLaplacePyr;													// 声明融合拉普拉斯金字塔
	Mat imgHighest;																	// 声明图像融合的起点图像
	Mat mask = Mat::zeros(PYRHEIGHT, PYRWIDTH, CV_32FC1);							// 构造掩码，大小与金字塔原图像相同
	mask(Range::all(), Range(mask.cols * threshold, mask.cols)) = 1.0;
	cvtColor(mask, mask, COLOR_GRAY2BGR);											// 将掩码颜色通道拓展，以适配原图像
	imgProcess::buildGaussPyr(mask, maskGaussPyr, 3);								// 建立掩码的高斯金字塔

	resize(leftImg, leftImg, Size(PYRWIDTH, PYRHEIGHT));
	resize(rightImg, rightImg, Size(PYRWIDTH, PYRHEIGHT));

	leftImg.convertTo(leftImg, CV_32F);		//转换成CV_32F, 用于和mask类型匹配,且CV_32F 类型精度高, 有利于计算
	rightImg.convertTo(rightImg, CV_32F);

	// 建立高斯金字塔与拉普拉斯金字塔
	imgProcess::buildGaussPyr(leftImg, gaussPyrLeft, 3);
	imgProcess::buildGaussPyr(rightImg, gaussPyrRight, 3);
	imgProcess::buildLaplacePyr(gaussPyrLeft, laplacePyrLeft, 3);
	imgProcess::buildLaplacePyr(gaussPyrRight, laplacePyrRight, 3);

	// 确定起点图像
	imgHighest  = gaussPyrLeft.back().mul(maskGaussPyr.back()) + 
		((gaussPyrRight.back()).mul(Scalar(1.0, 1.0, 1.0) - maskGaussPyr.back()));

	// 融合拉普拉斯金字塔
	imgProcess::blendLaplacePyr(laplacePyrLeft, laplacePyrRight, maskGaussPyr, blendLaplacePyr);

	// 融合图像重建
	dstImg = imgProcess::imgLaplaceBlend(imgHighest, blendLaplacePyr);
	dstImg.convertTo(dstImg, CV_8UC3);
	if (debug == DEBUGMODE_SHOW)	imshow("imgProcess::seamOpt_laplace", dstImg);
}
/*-----------------------------------------------------------------------------------*/


/*===================================================================================*/
/******************************* 私有函数 *********************************************/
/*===================================================================================*/

/*
 * @breif:建立高斯金字塔
 * @prama[in]:srcImg->待高斯金字塔化的源图像;imgPyr->金字塔化的图像集合;level->金字塔层数
 * @retval:None
 */
void imgProcess::buildGaussPyr(Mat srcImg, vector<Mat>& imgPyr, int level)
{
	imgPyr.push_back(srcImg);
	Mat tempImg;										// 存放采样的中间结果
	for (int i = 0; i < level; i++)
	{
		pyrDown(srcImg, tempImg, Size(srcImg.cols / 2, srcImg.rows / 2));
		imgPyr.push_back(tempImg);
		srcImg = tempImg;
	}
}

/*
 * @breif:建立拉普拉斯金字塔
 * @prama[in]:imgGaussPyr->图像的高斯金字塔;imgLaplacePyr->输出的图像拉普拉斯金字塔;level->金字塔层数
 * @retval:None
 */
void imgProcess::buildLaplacePyr(const vector<Mat> imgGaussPyr, vector<Mat>& imgLaplacePyr, int level)
{
	vector<Mat> imgGaussPyrCopy;
	Mat upLevel, downLevel, tempImg;

	imgGaussPyrCopy.assign(imgGaussPyr.begin(), imgGaussPyr.end());   // assign深拷贝
	for (int i = 0; i < level; i++)
	{
		Mat upLevel = imgGaussPyrCopy.back();						 // 获取高斯金字塔当前最高层图像
		imgGaussPyrCopy.pop_back();            
		downLevel = imgGaussPyrCopy.back();							 // 获取高斯金字塔次高层图像用于作差
		pyrUp(upLevel, tempImg, Size(upLevel.cols * 2, upLevel.rows * 2));
		imgLaplacePyr.push_back(downLevel- tempImg);
	}
	reverse(imgLaplacePyr.begin(), imgLaplacePyr.end());			 // 金字塔反转，使最底层面积最大
}

/*
 * @breif:建立融合拉普拉斯金字塔
 * @prama[in]:imgLp_1、imgLp_2->待融合图像的拉普拉斯金字塔;maskGauss->掩码的高斯金字塔(level+1层)
 * @prama[in]:blendLp->输出的融合拉普拉斯金字塔
 * @retval:None
 */
void imgProcess::blendLaplacePyr(const vector<Mat> imgLp_1, const vector<Mat> imgLp_2, const vector<Mat> maskGauss,
	vector<Mat>& blendLp)
{
	int level = imgLp_1.size();									// 金字塔层数

	for (int i = 0; i < level; i++)                      
	{
		Mat imgMask_1 = (imgLp_1.at(i)).mul(maskGauss.at(i));   // Mask加权 
		Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGauss[i];
		Mat imgMask_2 = imgLp_2[i].mul(antiMask);
		blendLp.push_back(imgMask_1+imgMask_2);					// 融合
	}
}

/*
 * @breif:图像拉普拉斯融合
 * @prama[in]:imgHighest->图像混合的起点,即两个待融合图像高斯金字塔最高层按mask加权求和的结果
 * @prama[in]:blendLp->输出的融合拉普拉斯金字塔
 * @retval:dstImg->融合的图像
 */
Mat imgProcess::imgLaplaceBlend(Mat& imgHighest, vector<Mat> blendLp)
{
	int level = blendLp.size();
	Mat upLevel, dstImg;
	for (int i = 0; i < level; i++)
	{
		pyrUp(imgHighest, upLevel, Size(imgHighest.cols * 2, imgHighest.rows * 2));
		dstImg = blendLp.back() + upLevel;
		blendLp.pop_back();
		imgHighest = dstImg;
	}
	return dstImg;
}
/*-----------------------------------------------------------------------------------*/