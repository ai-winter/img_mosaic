/*******************************************************************************
 *
 * \file    main.cpp
 * \brief   图像拼接主程序
 * \author  1851738杨皓冬+1853735赵祉淇
 * \version 5.0
 * \date    2021-06-17
 *
 * -----------------------------------------------------------------------------
 *
 * -----------------------------------------------------------------------------
 * 文件修改历史：
 * <时间>       | <版本>  | <作者>         |
 * 2021-06-08  | v1.0    | 1851738杨皓冬  |
 * 2021-06-09  | v2.0    | 1851738杨皓冬  |
 * 2021-06-11  | v3.0    | 1851738杨皓冬  |
 * 2021-06-12  | v4.0    | 1851738杨皓冬  |
 * 2021-06-12  | v5.0    | 1853735赵祉淇  |
 * -----------------------------------------------------------------------------
 ******************************************************************************/
#include "main.h"

int main(int argc, char* argv[])
{
    imgProcess imgProcessHandle("src\\imgfile.txt");        //加载图片
    Mat tempImgMosaic, tempImgHomo, dstImg;

    int mode(0);

    while (true)
    {
        cout << "请输入图像拼接的模式：1-SIFT, 2-ORB, 3-BRISK, 4-SURF, 0-QUIT" << endl;
        cin >> mode;

        if (mode == 1)
        {
            cout << "SIFT图像拼接结果如下 :" << endl;

            /*===================================================================================*/
            /******************************** 基于SIFT的图像拼接 ************************************/
            /*===================================================================================*/
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[1],
                imgProcessHandle.RGBImgs[2], SIFTDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            tempImgHomo = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, SIFTDETECT, MATCHMODE_MINMAX, DEBUGMODE_GETHOMO);
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, SIFTDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            imgProcessHandle.seamOpt_laplace(tempImgHomo, tempImgMosaic, dstImg, 0.35, DEBUGMODE_NORMAL);

            imshow("图像拼接", dstImg);
            waitKey(0);
        }
        else if (mode == 2)
        {
            cout << "ORB图像拼接结果如下 :" << endl;

            /*===================================================================================*/
            /******************************** 基于ORB的图像拼接 ************************************/
            /*===================================================================================*/
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[1],
                imgProcessHandle.RGBImgs[2], ORBDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            tempImgHomo = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, ORBDETECT, MATCHMODE_MINMAX, DEBUGMODE_GETHOMO);
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, ORBDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            imgProcessHandle.seamOpt_laplace(tempImgHomo, tempImgMosaic, dstImg, 0.35, DEBUGMODE_NORMAL);
            /*-----------------------------------------------------------------------------------*/

            imshow("图像拼接", dstImg);
            waitKey(0);
        }
        else if (mode == 3)
        {
            cout << "BRISK图像拼接结果如下 :" << endl;

            /*===================================================================================*/
            /******************************** 基于BRISK的图像拼接 **********************************/
            /*===================================================================================*/
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[1],
                imgProcessHandle.RGBImgs[2], BRISKDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            tempImgHomo = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, BRISKDETECT, MATCHMODE_MINMAX, DEBUGMODE_GETHOMO);
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, BRISKDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            imgProcessHandle.seamOpt_laplace(tempImgHomo, tempImgMosaic, dstImg, 0.35, DEBUGMODE_NORMAL);
            /*-----------------------------------------------------------------------------------*/

            imshow("图像拼接", dstImg);
            waitKey(0);
        }
        else if (mode == 4)
        {
            cout << "SURF图像拼接结果如下 :" << endl;

            /*===================================================================================*/
            /******************************** 基于SURF的图像拼接 ************************************/
            /*===================================================================================*/
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[1],
                imgProcessHandle.RGBImgs[2], SURFDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            tempImgHomo = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, SURFDETECT, MATCHMODE_MINMAX, DEBUGMODE_GETHOMO);
            tempImgMosaic = imageMosaic(imgProcessHandle, imgProcessHandle.RGBImgs[0],
                tempImgMosaic, SURFDETECT, MATCHMODE_MINMAX, DEBUGMODE_SHOW);
            imgProcessHandle.seamOpt_laplace(tempImgHomo, tempImgMosaic, dstImg, 0.35, DEBUGMODE_NORMAL);

            imshow("图像拼接", dstImg);
            waitKey(0);
        }
        else if (mode == 0)
            break;
        else
            cout << "输入无效，请重新输入" << endl;
    }

    return 0;
}


