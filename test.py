from cv2 import cv2
import numpy as np

# 显示图片
def show(img,winname='windows'):
    cv2.imshow(winname,img)
    cv2.waitKey()

# 标注个数
def num(img,contours,i=0):
    for contour in contours:
        # 二值化图轮廓中符合最优拟合椭圆条件
        if len(contour)>4:
            # 标注个数
            show(cv2.putText(img,str(i:=i+1),tuple(contour[1][0]),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,0),15))

# 主运行程序
def run(image):
    # 获取图像
    show(img := cv2.imdecode(np.fromfile(image,dtype=np.uint8),1))
    # 灰度图像
    show(img_gray := cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) 
    # 均衡化
    show(img_gray := cv2.equalizeHist(img_gray))
    # 二值化处理
    show(thresh := cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
    # 核函数
    kernel = np.ones((10,10),np.uint8)
    # 形态学腐蚀变换
    show(tmp := cv2.morphologyEx(thresh,cv2.MORPH_ERODE,kernel,iterations=10))
    # 距离变换
    show(imgs := cv2.distanceTransform(tmp,cv2.DIST_L2,3))
    # 阈值处理
    show(imgs := cv2.threshold(imgs,0.70*imgs.max(),255,0)[1])
    # 标注img图像
    num(img,cv2.findContours(np.uint8(imgs), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
    
# 测试图像
images = ['玉米颗粒.jpg']

if __name__=='__main__':
    # 界面设置
    cv2.namedWindow('windows',cv2.WINDOW_NORMAL)
    # 全屏显示
    cv2.setWindowProperty('windows',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # 读取图像参数
    from sys import argv
    from os.path import join
    images = argv[1:] if len(argv)>1 else (join('src',image) for image in images)
    # 尝试处理图像
    for image in images: 
        try:
            run(image)
        except FileNotFoundError:
            print(image,'不存在')
            continue
    