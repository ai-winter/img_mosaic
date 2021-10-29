#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include"ransac_personal.h"



//获得迭代次数
size_t GetIterationNumber(
	const float& inlier_ratio,
	const float& confidence,
	const size_t& sample_size
)
{
	float a = log(1.0 - confidence);
	float b = log(1.0 - std::pow(inlier_ratio, sample_size));

	if (abs(b) < std::numeric_limits<float>::epsilon())
		return std::numeric_limits<float>::epsilon();

	size_t it_num = static_cast<size_t>(a / b);

	return it_num;
}

//选择最小数量的随机样本点函数
void SelectMinimalSample
(
	size_t& n_points,
	std::vector<size_t>& sample,
	const size_t& k_sample_size
)
{
	sample.clear();
	// 1. Select a minimal sample -> 4 random points for homography
	while (sample.size() != k_sample_size)
	{
		// Generate a random index between [0, n_points]
		size_t random_idx =
			size_t((n_points - 1.0) * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

		if (std::find(sample.begin(), sample.end(), random_idx) == sample.end())
			sample.emplace_back(random_idx);
	}
}

//归一化坐标点函数
std::vector<cv::Point2f> NormalizePoints(
	std::vector<cv::Point2f>& points,
	std::vector<size_t>& indices,
	cv::Mat& translation,
	cv::Mat& scale
)
{
	std::vector<cv::Point2f> normalized_points;
	normalized_points.reserve(indices.size());
	// Calculate the mass point
	cv::Point2f mass_point(0, 0);
	translation = cv::Mat::eye(3, 3, CV_32F);
	scale = cv::Mat::eye(3, 3, CV_32F);

	for (const auto& idx : indices)
	{
		mass_point += points.at(idx);
		normalized_points.emplace_back(points.at(idx));
	}
	mass_point *= (1.0 / indices.size());
	// Move the points to have the origin in their mass point 
	// and calculate the average distance from the origin
	float avg_dist = 0.0;

	for (auto& p : normalized_points)
	{
		p -= mass_point;
		avg_dist += cv::norm(p);
	}
	avg_dist /= normalized_points.size();
	// Set up the translation matrix to unnormalize matrix_H later
	translation.at<float>(0, 2) = -mass_point.x;
	translation.at<float>(1, 2) = -mass_point.y;
	// Make the average distance for each point from the mass to be sqrt(2)
	const float ratio = sqrt(2) / avg_dist;
	for (auto& p : normalized_points)
		p *= ratio;
	// Set up the scale matrix to unnormalize matrix_H later
	scale.at<float>(0, 0) = ratio;
	scale.at<float>(1, 1) = ratio;

	return normalized_points;
}


//由4组对应的输入匹配点坐标生成计算式中的A矩阵
cv::Mat GetMatrixA(
	std::vector<cv::Point2f>& normalized_points_img1,
	std::vector<cv::Point2f>& normalized_points_img2,
	const size_t& k_sample_size
)
{
	cv::Mat matrix_A;

	for (size_t i = 0; i < k_sample_size; ++i)
	{
		cv::Mat vector = cv::Mat::zeros(2, 9, CV_32F);
		cv::Point2f& p1 = normalized_points_img1[i];
		cv::Point2f& p2 = normalized_points_img2[i];

		vector.at<float>(0, 0) = p1.x;
		vector.at<float>(0, 1) = p1.y;
		vector.at<float>(0, 2) = 1.0;
		vector.at<float>(0, 6) = -p1.x * p2.x;
		vector.at<float>(0, 7) = -p1.y * p2.x;
		vector.at<float>(0, 8) = -p2.x;

		vector.at<float>(1, 3) = p1.x;
		vector.at<float>(1, 4) = p1.y;
		vector.at<float>(1, 5) = 1.0;
		vector.at<float>(1, 6) = -p1.x * p2.y;
		vector.at<float>(1, 7) = -p1.y * p2.y;
		vector.at<float>(1, 8) = -p2.y;

		matrix_A.push_back(vector);
	}
	return matrix_A;
}

//计算投影次数矩阵
cv::Mat GetProjectionMatrix(cv::Mat& matrix_A)
{
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(matrix_A.t() * matrix_A, eigenvalues, eigenvectors);
	cv::Mat matrix_H = eigenvectors.row(eigenvectors.rows - 1);
	// Normalize matrix H
	matrix_H = matrix_H * (1.0 / matrix_H.at<float>(eigenvectors.rows - 1));

	return matrix_H;
}

//homo矩阵计算函数
cv::Mat CalculateHomographyMatrix(
	std::vector<cv::Point2f>& points_img1,
	std::vector<cv::Point2f>& points_img2,
	std::vector<size_t>& indices
)
{
	const size_t sample_size = indices.size();
	cv::Mat translation1, translation2, scale1, scale2;
	std::vector<cv::Point2f> normalized_points_img1 =
		NormalizePoints(points_img1, indices, translation1, scale1);
	std::vector<cv::Point2f> normalized_points_img2 =
		NormalizePoints(points_img2, indices, translation2, scale2);
	// Get matrix A formed from the selected n-n points
	cv::Mat matrix_A = GetMatrixA(normalized_points_img1,
		normalized_points_img2, sample_size);
	// Calculate the projection matrix H by getting the eigenvector
	// corresponding to the lowest eigenvalue of A^T*T and reshape it to 3x3
	cv::Mat matrix_H = GetProjectionMatrix(matrix_A);
	matrix_H = matrix_H.reshape(1, 3);

	cv::Mat T1 = scale1 * translation1;
	cv::Mat T2 = scale2 * translation2;
	//Unnormalize the Homography matrices
	matrix_H = T2.inv() * matrix_H * T1;
	// Check homography's correctness ---------------------------------------
	//checkHomographyCorrectness(normalized_points_img1, normalized_points_img2,
	//	matrix_H);
	// -----------------------------------------------------------------------
	// Check the unnormalized homography's correctness -----------------------
	//std::vector<cv::Point2d> selected_points1, selected_points2;
	//selected_points1.reserve(sample_indices.size());
	//selected_points2.reserve(sample_indices.size());
	//
	//for (const auto& idx : sample_indices)
	//{
	//	selected_points1.emplace_back(points_img1.at(idx));
	//	selected_points2.emplace_back(points_img2.at(idx));
	//}
	//checkHomographyCorrectness(selected_points1, selected_points2, matrix_H);
	// -----------------------------------------------------------------------
	return matrix_H;
}


//自定义的计算内含子算法
void CalculateInliers(
	std::vector<cv::Point2f>& points_img1,
	std::vector<cv::Point2f>& points_img2,
	cv::Mat& matrix_H,
	const float& threshold,
	std::vector<size_t>& current_inliers
)	//形参：两组匹配的特征点，计算得到的H矩阵，阈值参数，当前采样得到的inlier点集
{
	current_inliers.clear();	//清空当前内点集数据
	cv::Mat homogeneous_points1, homogeneous_points2;	//创建两个齐次mat矩阵

	cv::convertPointsToHomogeneous(points_img1, homogeneous_points1);
	cv::convertPointsToHomogeneous(points_img2, homogeneous_points2);	//将两个匹配点转换为其次形式
	cv::Mat matrix_H_inv = matrix_H.inv();	//对H求逆矩阵并赋值

	for (size_t idx = 0; idx < homogeneous_points1.rows; ++idx)
	{
		cv::Mat v1 = homogeneous_points1.row(idx).reshape(1, 3);
		cv::Mat v2 = homogeneous_points2.row(idx).reshape(1, 3);
		cv::Mat v1_prime = matrix_H * v1;
		cv::Mat v2_prime = matrix_H_inv * v2;

		float v1_prime_x = v1_prime.at<float>(0, 0) / v1_prime.at<float>(2, 0);
		float v1_prime_y = v1_prime.at<float>(1, 0) / v1_prime.at<float>(2, 0);
		float v2_prime_x = v2_prime.at<float>(0, 0) / v2_prime.at<float>(2, 0);
		float v2_prime_y = v2_prime.at<float>(1, 0) / v2_prime.at<float>(2, 0);

		cv::Point2f v1_prime_cartesian = { v1_prime_x, v1_prime_y };
		cv::Point2f v2_prime_cartesian = { v2_prime_x, v2_prime_y };

		float distance = cv::norm(points_img2.at(idx) - v1_prime_cartesian)
			+ cv::norm(points_img1.at(idx) - v2_prime_cartesian);

		if (distance < threshold)
		{
			current_inliers.emplace_back(idx);
		}
	}
}

//自定义的RANSAC计算homo矩阵算法（主体部分）
void GetHomographyRANSAC(
	std::vector<cv::Point2f>& points_img1,
	std::vector<cv::Point2f>& points_img2,
	const size_t& k_sample_size,
	cv::Mat& best_matrix_H,
	std::vector<size_t> best_inliers,
	const float& threshold,
	const size_t& max_iterations,
	const float& confidence
)
{
	// set random seed
	srand(time(NULL));
	size_t n_points = points_img1.size();
	// The current number of iterations
	size_t iteration_number = 0;
	size_t n_iterations = max_iterations;
	// The indices of the inliers of the current best model
	std::vector<size_t> current_inliers;
	current_inliers.reserve(points_img1.size());
	// The current sample indices
	std::vector<size_t> sample_indices;		//创建indices容器
	sample_indices.reserve(k_sample_size);	//动态调节采样样本的最大容量

	std::cout << "Searching for Homography with RANSAC!" << std::endl
		<< "Number of found point correspondences: " << points_img1.size()
		<< std::endl << "Threshold is: " << threshold << std::endl
		<< "Performing " << max_iterations << " iterations." << std::endl;

	while (iteration_number++ < n_iterations)	//迭代次数未达到限制时
	{
		if (iteration_number % 10 == 0)	//若迭代次数是10的倍数
			std::cout << "Current iteration: " << iteration_number << std::endl;	//输出当前迭代次数
		SelectMinimalSample(n_points, sample_indices, k_sample_size);	//从数据点中采集最小数量的数据点
		// collinearity check here....
		// Translation and Scale matrices
		cv::Mat matrix_H = CalculateHomographyMatrix(points_img1,
			points_img2, sample_indices);		//根据当前模型计算给定两组匹配的特征点之间的homo矩阵
		cv::Mat matrix_H_inv = matrix_H.inv();	//计算H矩阵的逆矩阵
		// Count the number of inliers
		CalculateInliers(points_img1, points_img2, matrix_H,
			threshold, current_inliers);	//计算当前状态下的内集合

		if (current_inliers.size() > best_inliers.size())	//当发现了当前最佳的内集合时输出最佳内集合的样本大小
		{
			std::cout << "Iteration number: " << iteration_number << std::endl
				<< "Current best inliers size: " << current_inliers.size()
				<< std::endl;

			best_inliers.swap(current_inliers);
			best_matrix_H = matrix_H.clone();	//复制当前的最佳homo矩阵
			current_inliers.clear();
			current_inliers.resize(0);			//清空所有缓冲区
		}
		// Update the maximum iteration number
		float inlier_ratio = static_cast<float>(best_inliers.size()) /
			static_cast<float>(points_img1.size());	//计算内集合率=内集合点数量/全部点数量
		n_iterations = GetIterationNumber(
			inlier_ratio,
			confidence,
			k_sample_size
		);	//更新最大迭代次数限制
	}
	best_matrix_H = CalculateHomographyMatrix(points_img1, points_img2,
		best_inliers);	//用最佳内集合计算对应的homo矩阵
}

//检查Homo矩阵是否正确
void checkHomographyCorrectness(
	std::vector<cv::Point2f>& points_img1,
	std::vector<cv::Point2f>& points_img2,
	cv::Mat& matrix_H
)
{
	cv::Mat homogeneous_points1, homogeneous_points2;
	std::vector<cv::Point3f> hom_proj_points1, hom_proj_points2;
	std::vector<cv::Point2f> inh_proj_points1, inh_proj_points2;

	inh_proj_points1.reserve(points_img1.size());
	inh_proj_points2.reserve(points_img2.size());

	cv::Mat matrix_H_inv = matrix_H.inv();

	cv::convertPointsToHomogeneous(points_img1, homogeneous_points1);
	cv::convertPointsToHomogeneous(points_img2, homogeneous_points2);

	for (size_t idx = 0; idx < homogeneous_points1.rows; ++idx)
	{
		cv::Mat v1 = homogeneous_points1.row(idx).reshape(1, 3);
		cv::Mat v2 = homogeneous_points2.row(idx).reshape(1, 3);

		cv::Mat v1_prime = matrix_H_inv * v2;
		cv::Mat v2_prime = matrix_H * v1;

		hom_proj_points1.emplace_back(cv::Point3f(v1_prime.at<float>(0, 0),
			v1_prime.at<float>(1, 0), v1_prime.at<float>(2, 0)));
		hom_proj_points2.emplace_back(cv::Point3f(v2_prime.at<float>(0, 0),
			v2_prime.at<float>(1, 0), v2_prime.at<float>(2, 0)));

	}
	cv::convertPointsFromHomogeneous(hom_proj_points1, inh_proj_points1);
	cv::convertPointsFromHomogeneous(hom_proj_points2, inh_proj_points2);

	for (size_t idx = 0; idx < points_img1.size(); ++idx)
	{
		std::cout << "v1: " << points_img1.at(idx) << std::endl << "v1_prime: "
			<< inh_proj_points1.at(idx) << std::endl << "v2: " << points_img2.at(idx)
			<< std::endl << "v2_prime" << inh_proj_points2.at(idx) << std::endl;
	}
}
