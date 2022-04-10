import numpy as np
import cv2

img = cv2.imread('raw.jpg')#호프변환을 수행할 이미지 불러오기
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#불러온 img를 이진변환
edges = cv2.Canny(imgray, 50, 150, apertureSize = 3)#인자값을 주어 캐니변환수행

#호프변환을 위한 인자값 마지막 인자는 threshold로 임계점이상의 엣지값 검출
lines = cv2.HoughLines(edges,1,np.pi/180,300)

for line in lines:#선검출을 위한 계산식 
    r, theta = line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*r
    y0=b*r
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*a)
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*a)

    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imwrite('hough.jpg',img)#호프변환이 된 결과이미지 생성 
   
