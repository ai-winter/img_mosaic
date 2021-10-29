/*******************************************************************************
 *
 * \file    featureMatch.cpp
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
 * 2021-06-11  | v1.0    | 1851738杨皓冬  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include "featureMatch.h"

 /*===================================================================================*/
 /******************************* 公有函数 *********************************************/
 /*===================================================================================*/

 /*
  * @breif:特征匹配，基于Low's算法
  * @prama[in]:Desc_1,Desc_2->待匹配图片的特征描述子,threshold->阈值,matchMode->匹配模式,宏定义
  * @retval:GoodMatchPoints->筛选出的优秀特征点匹配对
  */
vector<DMatch> featureMatch::featureMatch_Lows(const Mat Desc_1, const Mat Desc_2, float threshold, int matchMode)
{
    Mat smallDesc = featureMatch::getSmallDesc(Desc_1, Desc_2);
    Mat largeDesc = featureMatch::getLargeDesc(Desc_1, Desc_2);
    vector<DMatch> GoodMatchPoints;

    // 汉明距离度量
    if (matchMode == MATCHMODE_HAMMING)
    {
        Index flannIndex(smallDesc, LshIndexParams(12, 20, 2), featureMatch::matchModeTransFlann(matchMode));
        Mat matchIndex(largeDesc.rows, 2, CV_32SC1), matchDistance(largeDesc.rows, 2, CV_32FC1);
        flannIndex.knnSearch(largeDesc, matchIndex, matchDistance, 2, SearchParams());
        for (int i = 0; i < matchDistance.rows; i++)
        {
            if (matchDistance.at<float>(i, 0) < threshold * matchDistance.at<float>(i, 1))
            {
                DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
                GoodMatchPoints.push_back(dmatches);
            }
        }
    }

    // L2范数度量
    else if (matchMode == MATCHMODE_NORML2)
    {
        FlannBasedMatcher matcher;
        vector<vector<DMatch> > matchePoints;

        vector<Mat> train_desc(1, Desc_1);
        matcher.add(train_desc);
        matcher.train();
        matcher.knnMatch(Desc_2, matchePoints, 2);
        for (int i = 0; i < matchePoints.size(); i++)
        {
            if (matchePoints[i][0].distance < threshold * matchePoints[i][1].distance)
            {
                GoodMatchPoints.push_back(matchePoints[i][0]);
            }
        }
    }

    return GoodMatchPoints;
}

/*
 * @breif:特征匹配，基于minMax算法
 * @prama[in]:Desc_1,Desc_2->待匹配图片的特征描述子,threshold->阈值,matchMode->匹配模式,宏定义
 * @retval:GoodMatchPoints->筛选出的优秀特征点匹配对
 */
vector<DMatch> featureMatch::featureMatch_MinMax(const Mat Desc_1, const Mat Desc_2, float threshold, int matchMode)
{
    BFMatcher matcher(featureMatch::matchModeTransBFM(matchMode));          // 声明匹配器模式
    Mat smallDesc = featureMatch::getSmallDesc(Desc_1, Desc_2);             
    Mat largeDesc = featureMatch::getLargeDesc(Desc_1, Desc_2);
    vector<DMatch> matchPoints,GoodMatchPoints;

    matcher.match(smallDesc, largeDesc, matchPoints);
    sort(matchPoints.begin(), matchPoints.end());                           // 按照距离长短排序
    double minDist = matchPoints[0].distance;
    double maxDist = matchPoints[size(matchPoints) - 1].distance;

    for (int i = 0; i < smallDesc.rows; i++)
    {
        if (matchPoints[i].distance <= max(threshold * minDist, 30.0))  GoodMatchPoints.push_back(matchPoints[i]);
    }
    return GoodMatchPoints;
}

/*
 * @breif:获得优秀特征点对对应的原图像素坐标
 * @prama[in]:goodMatchPoints->筛选出的优秀特征点匹配对;keyPtLeft,keyPtRight->左右特征点集
 * @prama[in]:goodPtLeft,goodPtRight->左右优秀特征点
 * @retval:None
 */
void featureMatch::getGoodPt(vector<DMatch> goodMatchPoints, vector<KeyPoint> keyPtRight, vector<KeyPoint>keyPtLeft,
    vector<Point2f>&goodPtRight, vector<Point2f>&goodPtLeft)
{
    for (int i = 0; i < goodMatchPoints.size(); i++)
    {
        if (keyPtLeft.size() < keyPtRight.size())
        {
            goodPtLeft.push_back(keyPtLeft[goodMatchPoints[i].queryIdx].pt);
            goodPtRight.push_back(keyPtRight[goodMatchPoints[i].trainIdx].pt);
        }
        else
        {
            goodPtLeft.push_back(keyPtLeft[goodMatchPoints[i].trainIdx].pt);
            goodPtRight.push_back(keyPtRight[goodMatchPoints[i].queryIdx].pt);
        }
    }
}
/*-----------------------------------------------------------------------------------*/


/*===================================================================================*/
/******************************* 私有函数 *********************************************/
/*===================================================================================*/

/*
 * @breif:匹配模式转换为Flann、BFM
 * @prama[in]:matchMode->int格式匹配模式
 * @retval:matchMode->flann_distance_t或int格式匹配模式
 */
flann_distance_t featureMatch::matchModeTransFlann(int matchMode)
{
    switch (matchMode)
    {
    case(MATCHMODE_HAMMING):    return FLANN_DIST_HAMMING;
    case(MATCHMODE_NORML2):     return FLANN_DIST_L2;
    default:                    return FLANN_DIST_HAMMING;
        break;
    }  
}

int featureMatch::matchModeTransBFM(int matchMode)
{
    switch (matchMode)
    {
    case(MATCHMODE_HAMMING):    return NORM_HAMMING;
    case(MATCHMODE_NORML2):     return NORM_L2;
    default:                    return NORM_HAMMING;
        break;
    }
}

/*
 * @breif:获得二者中较小、较大的描述子
 * @prama[in]:Desc_1、Desc_2->特征描述子
 * @retval:smallDesc or largeDesc
 */
Mat featureMatch::getSmallDesc(const Mat Desc_1, const Mat Desc_2)
{
    if (Desc_1.rows > Desc_2.rows)  return Desc_2;
    else                            return Desc_1;
}

Mat featureMatch::getLargeDesc(const Mat Desc_1, const Mat Desc_2)
{
    if (Desc_1.rows > Desc_2.rows)  return Desc_1;
    else                            return Desc_2;
}
/*-----------------------------------------------------------------------------------*/