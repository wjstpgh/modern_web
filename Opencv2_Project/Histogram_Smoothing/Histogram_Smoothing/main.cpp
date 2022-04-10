#include "opencv2\highgui.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2\imgproc.hpp"
#include "opencv2\core.hpp"

using namespace std;
using namespace cv;


int main()
{
	Mat img_input, img_gray, img_result, img_histogram, img_histogram2;

	//이미지 파일을 읽어와서 img_input에 저장
	img_input = imread("ex3.png", IMREAD_COLOR);
	if (img_input.empty())
	{
		cout << "파일을 읽어올수 없습니다." << endl;
		exit(1);
	}


	//입력영상을 그레이스케일 영상으로 변환
	img_gray = Mat(img_input.rows, img_input.cols, CV_8UC1);

	for (int y = 0; y < img_input.rows; y++)
	{
		for (int x = 0; x < img_input.cols; x++)
		{
			//img_input으로부터 현재 위치 (y,x) 픽셀의
			//blue, green, red 값을 읽어온다. 
			uchar blue = img_input.at<Vec3b>(y, x)[0];
			uchar green = img_input.at<Vec3b>(y, x)[1];
			uchar red = img_input.at<Vec3b>(y, x)[2];

			//blue, green, red를 더한 후, 3으로 나누면 그레이스케일이 된다.
			uchar gray = (blue + green + red) / 3.0;

			//Mat타입 변수 img_gray에 저장한다. 
			img_gray.at<uchar>(y, x) = gray;
		}
	}


	//입력 그레이스케일 영상의 히스토그램 계산
	int histogram[256] = { 0, };

	for (int y = 0; y < img_input.rows; y++)
	{
		for (int x = 0; x < img_input.cols; x++)
		{
			int value = img_gray.at<uchar>(y, x);
			histogram[value] += 1;
		}
	}

	//입력 그레이스케일 영상의 누적 히스토그램 계산
	int cumulative_histogram[256] = { 0, };
	int sum = 0;

	for (int i = 1; i < 256; i++)
	{
		sum += histogram[i];
		cumulative_histogram[i] = sum;
	}

	//입력 그레이스케일 영상의 정규화된 누적 히스토그램 계산
	float normalized_cumulative_histogram[256] = { 0.0, };
	int image_size = img_input.rows * img_input.cols;

	for (int i = 0; i < 256; i++)
	{
		normalized_cumulative_histogram[i] = cumulative_histogram[i] / (float)image_size;
	}


	//히스토그램 평활화 적용 및 결과 영상의 히스토그램 계산
	img_result = Mat(img_input.rows, img_input.cols, CV_8UC1);
	int histogram2[256] = { 0, };
	for (int y = 0; y < img_input.rows; y++)
	{
		for (int x = 0; x < img_input.cols; x++)
		{
			img_result.at<uchar>(y, x) = normalized_cumulative_histogram[img_gray.at<uchar>(y, x)] * 255;
			histogram2[img_result.at<uchar>(y, x)] += 1;
		}
	}


	//결과 영상의 누적 히스토그램 계산
	int cumulative_histogram2[256] = { 0, };
	sum = 0;

	for (int i = 1; i < 256; i++)
	{
		sum += histogram2[i];
		cumulative_histogram2[i] = sum;
	}


	//히스토그램 그리기
	img_histogram = Mat(300, 600, CV_8UC1, Scalar(0));
	img_histogram2 = Mat(300, 600, CV_8UC1, Scalar(0));

	int max = -1;
	for (int i = 0; i < 256; i++)
		if (max < histogram[i]) max = histogram[i];

	int max2 = -1;
	for (int i = 0; i < 256; i++)
		if (max2 < histogram2[i]) max2 = histogram2[i];

	for (int i = 0; i < 256; i++)
	{
		int histo = 300 * histogram[i] / (float)max;
		int cumulative_histo = 300 * cumulative_histogram[i] / (float)cumulative_histogram[255];

		line(img_histogram, Point(i + 10, 300), Point(i + 10, 300 - histo), Scalar(255, 255, 255));
		line(img_histogram, Point(i + 300, 300), Point(i + 300, 300 - cumulative_histo), Scalar(255, 255, 255));


		int histo2 = 300 * histogram2[i] / (float)max2;
		int cumulative_histo2 = 300 * cumulative_histogram2[i] / (float)cumulative_histogram2[255];

		line(img_histogram2, Point(i + 10, 300), Point(i + 10, 300 - histo2), Scalar(255, 255, 255));
		line(img_histogram2, Point(i + 300, 300), Point(i + 300, 300 - cumulative_histo2), Scalar(255, 255, 255));
	}


	//화면에 결과 이미지를 보여준다.
	imshow("입력 영상", img_input);
	imshow("입력 그레이스케일 영상", img_gray);
	imshow("결과 그레이스케일 영상", img_result);
	imshow("입력 영상의 히스토그램", img_histogram);
	imshow("평활화 후 히스토그램", img_histogram2);

	//아무키를 누르기 전까지 대기
	while (cv::waitKey(0) == 0);

	//결과를 파일로 저장
	imwrite("img_gray.jpg", img_gray);
	imwrite("img_result.jpg", img_result);
	imwrite("img_histogram.jpg", img_histogram);
	imwrite("img_histogram2.jpg", img_histogram2);
}