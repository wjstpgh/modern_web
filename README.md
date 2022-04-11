# Univ_Opencv

## 히스토그램 평활화

영상물에는 화소라는 단위가 있다. 그 화소들의 명암 값을 추출하여 히스토그램 그래프를 작성하는 원리를 이해하고, 나아가 히스토그램 인자들의 값을 조절하여 흑백사진의 화소들의 명암 분포를 균일하게 재배치하여 밝기 분포를 재배치하는 원리를 이해하는 것을 목표로 하여 opencv의 자체코드를 이해하고 평활화 코드를 자체 제작하여 서로 비교하는 것이 목표이다.

![image](https://user-images.githubusercontent.com/26988563/162623521-7d44bfd7-5635-410f-a881-c5092bd4bf38.png)

opencv 히스토그램 평활화 코드로 대상 이미지를 받아와 커맨드라인을 사용해 open을 해준 후 mat을 사용하여 화소들을 읽어들인다. 후에 오류처리를 해주고 cvt를 사용하여 원 명암값을 equalizehist함수를 통하여 재배치하여 결과물 dst로 저장하여 원본과 재배치된 결과를 이미지로 띄워주는 코드이다. 아래의 커맨드창은 이미지파일이 없을 때 오류처리가 된 결과를 보여주고 있다.

* Result

![image](https://user-images.githubusercontent.com/26988563/162623526-5f0a9d23-b01b-4ebb-9742-9bb868cd01f8.png)

![image](https://user-images.githubusercontent.com/26988563/162623528-17fc1300-892a-45d3-9cc1-4a4d623f3d19.png)

![image](https://user-images.githubusercontent.com/26988563/162623534-a9a957d4-8ff2-40f1-a2b2-a03c4f3e11aa.png)

* Make code

```
#include <stdio.h>
#include <iostream>
#include "opencv2\imgproc.hpp"
#include "opencv2\core.hpp"
#include "opencv2\highgui.hpp"

using namespace std;
using namespace cv;


int main()
{
	Mat raw_img, gray_PIX, result_img, histo_row, histo_colmn;

	raw_img = imread("ex3.png", IMREAD_COLOR);
	if (raw_img.empty())//예외처리
	{
	cout << "FILE ERROR!!" << endl;
	exit(1);
	}

	gray_PIX = Mat(raw_img.rows, raw_img.cols, CV_8UC1);

	for (int y = 0; y < raw_img.rows; y++)//화소값 배열에 지정
	{
	for (int x = 0; x < raw_img.cols; x++)
	{
	gray_PIX.at<uchar>(y, x) = raw_img.at<uchar>(y, x);
	}
	}

	//입력 그레이스케일 영상의 히스토그램 계산
	int histogram[256] = { 0, };

	for (int y = 0; y < raw_img.rows; y++)
	{
	for (int x = 0; x < raw_img.cols; x++)
	{
	int value = gray_PIX.at<uchar>(y, x);
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
	int image_size = raw_img.rows * raw_img.cols;

	for (int i = 0; i < 256; i++)
	{
	normalized_cumulative_histogram[i] = cumulative_histogram[i] / (float)image_size;
	}

	//히스토그램 평활화 적용 및 결과 영상의 히스토그램 계산
	result_img = Mat(raw_img.rows, raw_img.cols, CV_8UC1);
	int histogram2[256] = { 0, };
	for (int y = 0; y < raw_img.rows; y++)
	{
	for (int x = 0; x < raw_img.cols; x++)
	{
	result_img.at<uchar>(y, x) = normalized_cumulative_histogram[gray_PIX.at<uchar>(y, x)] * 255;
	histogram2[result_img.at<uchar>(y, x)] += 1;
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
	histo_row = Mat(300, 600, CV_8UC1, Scalar(0));
	histo_colmn = Mat(300, 600, CV_8UC1, Scalar(0));

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

	line(histo_row, Point(i + 10, 300), Point(i + 10, 300 - histo), Scalar(255, 255, 255));
	line(histo_row, Point(i + 300, 300), Point(i + 300, 300 - cumulative_histo), Scalar(255, 255, 255));


	int histo2 = 300 * histogram2[i] / (float)max2;
	int cumulative_histo2 = 300 * cumulative_histogram2[i] / (float)cumulative_histogram2[255];

	line(histo_colmn, Point(i + 10, 300), Point(i + 10, 300 - histo2), Scalar(255, 255, 255));
	line(histo_colmn, Point(i + 300, 300), Point(i + 300, 300 - cumulative_histo2), Scalar(255, 255, 255));
	}


	//화면에 결과 이미지를 보여준다.
	imshow("입력 영상", raw_img);
	imshow("입력 그레이스케일 영상", gray_PIX);
	imshow("결과 그레이스케일 영상", result_img);
	imshow("입력 영상의 히스토그램", histo_row);
	imshow("평활화 후 히스토그램", histo_colmn);

	//아무키를 누르기 전까지 대기
	while (cv::waitKey(0) == 0);

	//결과를 파일로 저장
	imwrite("img_gray.jpg", gray_PIX);
	imwrite("img_result.jpg", result_img);
	imwrite("img_histogram.jpg", histo_row);
	imwrite("img_histogram2.jpg", histo_colmn);
}
```

* Result

![image](https://user-images.githubusercontent.com/26988563/162623542-f38edd17-52bc-4a1a-bc8a-2a1b8a7bbe4f.png)

![image](https://user-images.githubusercontent.com/26988563/162623546-be4cff5a-6777-41f2-ab87-833e8376c214.png)

![image](https://user-images.githubusercontent.com/26988563/162623548-653c093d-5658-436b-8c25-2ea400477937.png)

히스토그램 평활화를 두 가지 코드로 작성해서 실행해 보았다. 우선 opencv의 함수를 사용하여 평활화를 프로그램으로 어떻게 수행하는지에 대한 대략적인 감을 잡고 직접 코드로 작성해보았다. 사실상 opencv에서 패키지로 링크되었던 함수들을 끄집어내어 작성한 것이다. 개념자체는 어렵지 않았다. 이미지를 단위별로 읽어들여 일정한 연산과정을 거쳐 연산된 결과들을 배치하여 새로운 이미지파일을 만들어내는 과정이었다. 실제 수행과정과 결과는 별 차이가 없지만 차이점이 있다면 방법의 차이이다. 아무래도 클래스화를 못하다 보니 코드들을 수행하는데에 있어 cpu와 메모리 관점에서 수행속도나 효율성이 떨어졌다고 생각한다.

## Hough transform

호프변환의 개념을 이해하고 실제 코드를 제작한 후 opencv에서 제공하는 함수와 비교함으로써 호프 변환을 실습해본다.

> 호프변환이란
> * 이미지 상에 존재하는 직선을 함수의 개념으로 나타낸다.
> * 직선을 xy가 아닌 다른 변수로 다른 평면에서 나타낸다.
> * 이떄 다른 변수는 유한하게 나타낼 수 있는 원점에서의 거리와 각도로 정한다.
> * 나타낸 값들의 집합으로 직선이 존재할 확률에 따라 직선의 존재 유무를 확률로써 판별한다.
> * 이러한 개념으로 엣지 검출 개념을 정한다.

* Opencv code

```
import numpy as np
import cv2

img = cv2.imread('raw1.jpg')#호프변환을 수행할 이미지 불러오기
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#불러온 img를 이진변환
edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)#인자값을 주어 캐니변환수행

#호프변환을 위한 인자값 마지막 인자는 threshold로 임계점이상의 엣지값 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)

for line in lines : #선검출을 위한 계산식
r, theta = line[0]
a = np.cos(theta)
b = np.sin(theta)
x0 = a*r
y0 = b*r
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * a)
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * a)

cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imwrite('hough.jpg', img)#호프변환이 된 결과이미지 생성
```

* result

![image](https://user-images.githubusercontent.com/26988563/162659074-f99aa765-f2fb-4b97-9224-d32594562149.png)

![image](https://user-images.githubusercontent.com/26988563/162659082-beb7ea9c-698c-4cb8-8024-7b11dac2b512.png)

![image](https://user-images.githubusercontent.com/26988563/162659089-63217c1a-25e7-48a2-b5e8-9892159bc059.png)

두 번째와 세 번째의 결과창은 동일한 raw이미지로 변환을 시행했지만 처음 변환을 시도했을 시에 임계점 이상으로 검출되는 엣지가 너무 많이 나와서 임계값을 조절해야 한다. 이렇듯 이미지에 따라 임계값을 조절해주면 원하는 강도만큼의 엣지를 검출하는 것을 조절할 수 있다.

![image](https://user-images.githubusercontent.com/26988563/162659149-36c621eb-f2f6-4570-a086-5a8cc408bb03.png)

위에 작성되었듯이 임계값이 너무 높아 이미지에서 출력할 엣지가 없는 경우에는 아예 결과창이 생성되지 않는다. 위와 같이 엣지가 검출되었을 때에도 임계값을 낮춘다면 더 많은 선들을 볼 수 있다.

* Make code

```
#define BITMAP_WIDTH 800
#define BITMAP_HEIGHT 600

#define IMAGE_DIAGONAL sqrtf(BITMAP_WIDTH*BITMAP_WIDTH + BITMAP_HEIGHT*BITMAP_HEIGHT) //대각선의 길이

#define thta 270
#define line_thrhold 40
#define thrhold 5

void HoughTransform(LPB* out_Image, LPBYTE* inputImage, int nW, int nH)
{
	register int i, j, k, l, m;
	int d;
	float p2d = 3.14f / 180.0f;

	int thres = 20;

	for (i = 0;i < IMAGE_DIAGONAL;i++)
	{
		for (j = 0;j < thta;j++)
			H[i][j] = 0;
	}

	float* LUT_COS = new float[thta];
	float* LUT_SIN = new float[thta];
	float* LUT_COT = new float[thta];
	float* LUT_SEC = new float[thta];

	for (i = 0;i < thta;i++)
	{
		LUT_COS[i] = cosf((i)*p2d);
		LUT_SIN[i] = sinf((i)*p2d);
		LUT_COT[i] = 1 / tanf(i * p2d);
		LUT_SEC[i] = 1 / LUT_SIN[i];
	}

	for (i = 0;i < nH;i++)
		for (j = 0;j < nW;j++)
			if (inputImage[i][j] > thrhold)
			{
				for (k = 0;k < thta;k++)
				{
					d = (int)(i * LUT_COS[k] + j * LUT_SIN[k]);

					if (d >= 0 && d < IMAGE_DIAGONAL)
						H[d][k] += 1;
				}
			}
}
 }
 for (i = 0;i < nH;i++)
 {
	 for (j = 0;j < nW;j++)
		 outImage[i][j] = 0;
 }
 int w = 0;
 int max_w = w;
 float theta_thres = 0.1;
 double const PI = 3.1415926535;
 for (d = 0;d < IMAGE_DIAGONAL;d++)
 {
	 for (k = 0;k < thta;k++)
	 {
		 if (H[d][k] > thres)
		 {
			 int max = H[d][k];

			 for (l = -4;l < 4;l++)
			 {
				 for (m = -4;m < 4;m++)
				 {
					 if (d + l >= 0 && d + l < IMAGE_DIAGONAL && k + m >= 0 && k + m < 180)
					 {
						 if (H[d + l][k + m] > max)
							 max = H[d + l][k + m];
					 }
				 }
			 }

			 if (max > H[d][k]) continue;
			 {
				 for (j = 0;j < nW;j++)
				 {
					 i = (int)((d - j * LUT_SIN[k]) / LUT_COS[k]);

					 if (i < nH && i > 0)
					 {
						 if (inputImage[i][j] > thrhold)
							 w++;
						 else
						 {
							 if (w > max_w)
								 max_w = w;
							 w = 0;
						 }
					 }
				 }
				 if (max_w > line_thrhold)
				 {
					 for (j = 0;j < nW;j++)
					 {
						 i = (int)((d - j * LUT_SIN[k]) / LUT_COS[k]);

						 if (i < nH && i >= 0)
						 {
							 outImage[i][j] += 1;
						 }
					 }
				 }
			 }
			 {
				 w = 0;
				 max_w = w;
				 for (i = 0;i < nH;i++)
				 {
					 j = (int)((d - i * LUT_COS[k]) / LUT_SIN[k]);

					 if (j < nW && j > 0)
					 {
						 if (inputImage[i][j] > thrhold)
							 w++;
						 else
						 {
							 if (w > max_w)
								 max_w = w;
							 w = 0;
						 }
					 }
				 }
				 if (max_w > line_thrhold)
				 {
					 for (i = 0;i < nH;i++)
					 {
						 j = (int)((d - i * LUT_COS[k]) / LUT_SIN[k]);

						 if (j < nW && j >= 0)
						 {
							 //if(inputImage[i][j] > SHARP_THRESHOLD)
							 outImage[i][j] += 1;

						 }

					 }
				 }
			 }
		 }
	 }
 }
 delete[]LUT_COS;
 delete[]LUT_SIN;
 delete[]LUT_COT;
 delete[]LUT_SEC;
}
```

* result

![image](https://user-images.githubusercontent.com/26988563/162659357-7c40720f-c365-4616-8d9c-4434c0b9ca05.png)

![image](https://user-images.githubusercontent.com/26988563/162659359-877cf3d2-680b-4ffc-b7e2-8369d00e30ef.png)

![image](https://user-images.githubusercontent.com/26988563/162659370-3eb8458c-f73b-4bbf-b746-baeef114391e.png)

임계값을 300으로 주었을 때 자체제작 코드의 결과 창들이다.

자체제작코드와 opencv함수를 사용했을 때의 차이는 임계값을 같게 했을 때의 차이는 없었다. 사실 실행시간에도 별 차이가 없었고 다만 opencv에서의 함수를 풀어서 직접 작성하다보니 코드가 길어진 것에 대해 차이가 있었다고 생각한다.




