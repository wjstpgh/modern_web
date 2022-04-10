
import cv2 #opencv 패키지 import

def show(): #이미지파일을 읽는 함수정의
    imgfile = 'model.jpg' #출력할 이미지의 경로 지정
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    #파일경로와 읽는 방식을 인자로 받아서 객체를 리턴해줌

    cv2.imshow('model', img)#img 객체를 화면에 나타내주는 함수
    cv2.waitKey(0) #키보드 입력값이 있을 때까지 대기시키는 함

show()
