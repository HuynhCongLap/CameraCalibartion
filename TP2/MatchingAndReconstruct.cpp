#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <fstream>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;



Mat triangulate_Linear_LS(Mat mat_P_l, Mat mat_P_r, Mat warped_back_l, Mat warped_back_r)
{
	Mat A(4, 3, CV_64FC1), b(4, 1, CV_64FC1), X(3, 1, CV_64FC1), X_homogeneous(4, 1, CV_64FC1), W(1, 1, CV_64FC1);
	W.at<double>(0, 0) = 1.0;
	A.at<double>(0, 0) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(0, 0);
	A.at<double>(0, 1) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(0, 1);
	A.at<double>(0, 2) = (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(0, 2);
	A.at<double>(1, 0) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 0) - mat_P_l.at<double>(1, 0);
	A.at<double>(1, 1) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 1) - mat_P_l.at<double>(1, 1);
	A.at<double>(1, 2) = (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 2) - mat_P_l.at<double>(1, 2);
	A.at<double>(2, 0) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(0, 0);
	A.at<double>(2, 1) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(0, 1);
	A.at<double>(2, 2) = (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(0, 2);
	A.at<double>(3, 0) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 0) - mat_P_r.at<double>(1, 0);
	A.at<double>(3, 1) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 1) - mat_P_r.at<double>(1, 1);
	A.at<double>(3, 2) = (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 2) - mat_P_r.at<double>(1, 2);
	b.at<double>(0, 0) = -((warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(0, 3));
	b.at<double>(1, 0) = -((warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0))*mat_P_l.at<double>(2, 3) - mat_P_l.at<double>(1, 3));
	b.at<double>(2, 0) = -((warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(0, 3));
	b.at<double>(3, 0) = -((warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0))*mat_P_r.at<double>(2, 3) - mat_P_r.at<double>(1, 3));
	solve(A, b, X, DECOMP_SVD);
	vconcat(X, W, X_homogeneous);
	return X_homogeneous;
}
int main()
{
	
	cv::FileStorage opencv_file("input.yml", cv::FileStorage::READ); // read file

	cv::Mat camera_matrix; // Camera matrix 3x3
	cv::Mat extrinsic; // vector ratation and translation

	opencv_file["camera_matrix"] >> camera_matrix; // get camera matrix from file
	opencv_file["extrinsic_parameters"] >> extrinsic; // get extrinsic vector from file
	
	cout << camera_matrix<<endl;
	cout << extrinsic <<endl;
	
	extrinsic.convertTo(extrinsic, CV_64F);
	vector <Mat> project; // projection matrix every frame

	for (int i = 0; i < extrinsic.rows; i++)
	{
		Mat row = extrinsic.row(i);
		Mat ro = cv::Mat(3, 1, CV_64F); // matrix rotation of this frame
		Mat t = cv::Mat(3, 1, CV_64F); // matrix translation of this frame

		ro.at<double>(0, 0) = row.at<double>(0,0);
		ro.at<double>(1, 0) = row.at<double>(0,1);
		ro.at<double>(2, 0) = row.at<double>(0,2);

		t.at<double>(0, 0) = row.at<double>(0,3);
		t.at<double>(1, 0) = row.at<double>(0,4);
		t.at<double>(2, 0) = row.at<double>(0,5);

		Mat projection;
		Rodrigues(ro, projection); // convert from 3x1 rotation to 3x3 ratation matrix
		
		cv::hconcat(projection, t, projection); // create projection matrix
		projection = camera_matrix * projection; // K * [R|t]
		project.push_back(projection);

		//Mat rotation;
		//Mat translate;
		//cout << "------------" << endl;
		//cout <<project[i] << endl;
	}
	//cout << project.size() << endl;
	opencv_file.release();

	int NB_Images = 13;
	ofstream myfile("example.txt");

	
	for (int ii = 0; ii < NB_Images; ii++) {
		for (int jj = ii+1; jj < NB_Images; jj++) {

			string img1 = format("View_%d.jpg", ii);
			string img2 = format("View_%d.jpg", jj);

			Mat img_1 = imread(img1, IMREAD_COLOR);
			Mat img_2 = imread(img2, IMREAD_COLOR);

			if (!img_1.data || !img_2.data)
			{
				std::cout << " --(!) Error reading images " << std::endl;
				waitKey();
				return -1;
			}
			//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
			int minHessian = 400;
			Ptr<SURF> detector = SURF::create();
			detector->setHessianThreshold(minHessian);
			std::vector<KeyPoint> keypoints_1, keypoints_2;
			Mat descriptors_1, descriptors_2;
			detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
			detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);
			//-- Step 2: Matching descriptor vectors using FLANN matcher
			FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
			matcher.match(descriptors_1, descriptors_2, matches);
			double max_dist = 0; double min_dist = 100;
			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptors_1.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);
			//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
			//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
			//-- small)
			//-- PS.- radiusMatch can also be used here.
			std::vector< DMatch > good_matches;
			for (int i = 0; i < descriptors_1.rows; i++)
			{

				if (matches[i].distance <= max(2 * min_dist, 0.02))
				{
					good_matches.push_back(matches[i]);
				}
			}
			//-- Draw only "good" matches
			Mat img_matches;
			drawMatches(img_1, keypoints_1, img_2, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Show detected matches
			imshow("Good Matches", img_matches);

			vector <Mat> left_point;
			vector <Mat> right_point;
			for (int i = 0; i < (int)good_matches.size(); i++)
			{
				printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
			}

			for (int i = 0; i < (int)good_matches.size(); i++)
			{
				cout << keypoints_1[good_matches[i].queryIdx].pt << endl;
				cout << keypoints_2[good_matches[i].trainIdx].pt << endl;

				Mat point1 = cv::Mat(3, 1, CV_64F);
				point1.at<double>(0, 0) = keypoints_1[good_matches[i].queryIdx].pt.x;
				point1.at<double>(1, 0) = keypoints_1[good_matches[i].queryIdx].pt.y;
				point1.at<double>(2, 0) = 1;
				left_point.push_back(point1);

				Mat point2 = cv::Mat(3, 1, CV_64F);
				point2.at<double>(0, 0) = keypoints_2[good_matches[i].trainIdx].pt.x;
				point2.at<double>(1, 0) = keypoints_2[good_matches[i].trainIdx].pt.y;
				point2.at<double>(2, 0) = 1;
				right_point.push_back(point2);
			}

			cout << "Number of point: " << left_point.size() << endl;
			cout << "Number of point: " << right_point.size() << endl;




			for (int i = 0; i < left_point.size(); i++)
			{
				Mat save = triangulate_Linear_LS(project[ii], project[jj], left_point[i], right_point[i]);
				if (myfile.is_open())
				{
					myfile << save.at<double>(0, 0) << " " << save.at<double>(1, 0) << " " << save.at<double>(2, 0) << endl;
				}

			}
		}

	}
	
	myfile.close();
	waitKey(0);
	
	int n;
	cin >> n;
	return 0;
}
