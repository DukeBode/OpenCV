from cv2 import cv2
import numpy as np
from os.path import join,abspath

# 图像
images = ['药物颗粒.jpg','玉米颗粒.jpg']

image = abspath(join('src',images[1]))
# 显示图片
def show(img,winname='windows',x=900,y=700):
    cv2.imshow(winname,cv2.resize(img,(x,y)))
    cv2.imshow(winname,img)
    cv2.waitKey()

# 界面设置
cv2.namedWindow('windows',0)
cv2.moveWindow('windows',0,0)

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
# 计数置零
i=0
# 从二值化图轮廓中最优拟合椭圆
for contour in cv2.findContours(np.uint8(imgs), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
    if len(contour)>4:
        i= i+1
        cv2.ellipse(img,cv2.fitEllipse(contour),(0,255,255),3)
        show(img)
print(i)
