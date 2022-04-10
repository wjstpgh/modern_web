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

			 /*
			   윤곽선을 검출을 하게되면 실제로 두껍게 감지되기 때문에,
			   직선도 여러줄이 감지가 되게 된다.
			   여러줄이 감지가 되는것을 방지하고 감지된 직선을 얇게 하기 위해서 밑의 코드를 삽입
			 */
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
			 //if((k <= 45 && k <= 315) || (k >= 135 && k <= 225))
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
							 //if(inputImage[i][j] > SHARP_THRESHOLD)
							 outImage[i][j] += 1;
						 }
					 }
				 }
			 }


			 //if(LUT_COT[k] <= 3.141592/4 && LUT_COT[k] >= 3.141592*5/4)
			 //else
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