/*******************************************************************************
 *
 * \file    main.h
 * \brief   图像拼接主程序
 * \author  1851738杨皓冬
 * \version 3.0
 * \date    2021-06-17
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-11  | v1.0    | 1851738杨皓冬  |
 * 2021-06-12  | v2.0    | 1851738杨皓冬  |
 * 2021-06-17  | v3.0    | 1853735赵祉淇  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include "imgProcess.h"
#include "homoEstimation.h"
#include "featureDesc.h"
#include "featureMatch.h"

#pragma once
#ifndef MAIN_H
#define MAIN_H

/*
 * @breif:图像拼接与美化
 * @prama[in]:handle->图像处理句柄;leftImg->待拼接的左图;rightImg->待拼接的右图;
 * @prama[in]:detectMode->检测模式(SIFT、ORB、BRISK等)
 * @prama[in]:matchType->匹配类型(minmax算法或low's算法)
 * @prama[in]:debug->调试模式
 * @retval:mosaicImg->由leftImg、rightImg拼接而成的图像
 */
Mat imageMosaic(imgProcess handle, Mat leftImg, Mat rightImg,int detectMode, int matchType, int debug = DEBUGMODE_SHOW)
{
    /*===================================================================================*/
    /******************************** 声明句柄、资源等 **************************************/
    /*===================================================================================*/
    featureDesc featureDescHandle;                          // 声明特征描述句柄
    featureMatch featureMatchHandle;                        // 声明特征匹配句柄
    MatSize imgSize = rightImg.size;                        // 声明拼接图像尺寸
    vector<KeyPoint> keyPtRight, keyPtLeft;                 // 声明关键点
    Mat imgDescRight, imgDescLeft;                          // 声明描述子
    vector<DMatch> goodMatchPt;                             // 声明优秀匹配点对
    vector<Point2f> goodPtLeft, goodPtRight;                // 声明优秀匹配点
    Mat grayImgLeft, grayImgRight;                          // 声明灰度图
    cvtColor(leftImg, grayImgLeft, COLOR_RGB2GRAY);
    cvtColor(rightImg, grayImgRight, COLOR_RGB2GRAY);
    /*-----------------------------------------------------------------------------------*/


    /*===================================================================================*/
    /******************************** 特征检测、描述与匹配 ***********************************/
    /*===================================================================================*/
    if (detectMode == SIFTDETECT)
    {
        featureDescHandle.getFeatureDesc_SIFT(grayImgLeft, keyPtLeft, imgDescLeft);
        featureDescHandle.getFeatureDesc_SIFT(grayImgRight, keyPtRight, imgDescRight);
        goodMatchPt = featureMatchHandle.featureMatch_MinMax(imgDescLeft, imgDescRight, 2, MATCHMODE_NORML2);
    }
    else if (detectMode == SURFDETECT)
    {
        featureDescHandle.getFeatureDesc_SURF(grayImgLeft, keyPtLeft, imgDescLeft);
        featureDescHandle.getFeatureDesc_SURF(grayImgRight, keyPtRight, imgDescRight);
        goodMatchPt = featureMatchHandle.featureMatch_MinMax(imgDescLeft, imgDescRight, 2, MATCHMODE_NORML2);
    }
    else if (detectMode == ORBDETECT)
    {
        featureDescHandle.getFeatureDesc_ORB(grayImgLeft, keyPtLeft, imgDescLeft);
        featureDescHandle.getFeatureDesc_ORB(grayImgRight, keyPtRight, imgDescRight);
        if (matchType)
            goodMatchPt = featureMatchHandle.featureMatch_MinMax(imgDescLeft, imgDescRight, 2.4, MATCHMODE_HAMMING);
        else
            goodMatchPt = featureMatchHandle.featureMatch_Lows(imgDescLeft, imgDescRight, 0.5, MATCHMODE_HAMMING);
    }
    else if (detectMode == BRISKDETECT)
    {
        featureDescHandle.getFeatureDesc_BRISK(grayImgLeft, keyPtLeft, imgDescLeft);
        featureDescHandle.getFeatureDesc_BRISK(grayImgRight, keyPtRight, imgDescRight);
        goodMatchPt = featureMatchHandle.featureMatch_MinMax(imgDescLeft, imgDescRight, 2.3, MATCHMODE_HAMMING);
    }
    featureMatchHandle.getGoodPt(goodMatchPt, keyPtRight, keyPtLeft, goodPtRight, goodPtLeft);
    //++++

    if (debug == DEBUGMODE_GETMATCH)
    {
        Mat imgMatch;
        drawMatches(leftImg, keyPtLeft, rightImg, keyPtRight, goodMatchPt, imgMatch, Scalar(0, 255, 255));
        return imgMatch;
    }
    /*-----------------------------------------------------------------------------------*/


    /*===================================================================================*/
    /************************************ 单应性估计 ***************************************/
    /*===================================================================================*/
    homoEst homographyMap(goodPtRight, goodPtLeft, imgSize);         // 以左图为基准,右图映射到左图
    homographyMap.findHomography_Base();    //++++change++++
    homographyMap.calTransBound();
    Size mapSize = Size(homographyMap.rightBound, imgSize[0]);       // 映射图片大小
    Mat imgMapByHomo = homographyMap.imgMapByHomo(rightImg, homographyMap.H, mapSize);
    if (debug == DEBUGMODE_GETHOMO)  return imgMapByHomo;
    /*-----------------------------------------------------------------------------------*/


    /*===================================================================================*/
    /************************************ 图像配准与美化 ***********************************/
    /*===================================================================================*/
    Mat dstImg = handle.imgMosaic(leftImg, imgMapByHomo);
    if (debug == DEBUGMODE_GETMOSAIC)   return dstImg;
    handle.seamOpt_alpha(leftImg, imgMapByHomo, dstImg, homographyMap.leftBound, imgSize[1]);
    if(debug==DEBUGMODE_SHOW)           return dstImg;
    /*-----------------------------------------------------------------------------------*/
}

#endif // !MAIN_H
