#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:33:47 2018

@author: guoxiaowen
"""

import sys
import os
from captcha.image import ImageCaptcha
import random 
import time
import shutil


#  生成验证码的数据集
CHAR_SET  = ['0','1','2','3','4','5','6','7','8','9']

# 验证码生成的数据集长度
CHAR_SET_LEN = 10


#  生成的验证码的长度
CAPTCHA_LEN = 4

#   训练验证码的保存位置
CAPTCHA_IMAGE_PATH = "/Volumes/study/2018大三上课程/大数据分析/作业/train_picture/"

#   测试训练验证码的保存位置
TEST_IMAGE_PATH = "/Volumes/study/2018大三上课程/大数据分析/作业/test_picture/"

#   测试验证码的数量
TEST_IMAGE_NUMBER = 50

#   按照顺序生成验证码，并进行保存

def generate_captcha_image(CharSet = CHAR_SET,CharSetLen = CHAR_SET_LEN,CaptchaImgPath = CAPTCHA_IMAGE_PATH):
    
    count = 0
    
#  生成随机的验证码  0000-9999        
    for i in range(CharSetLen):
        for j in range(CharSetLen):
            for k in range(CharSetLen):
                for m in range(CharSetLen):
                    captcha_text = CharSet[i] + CharSet[j] + CharSet[k] + CharSet[m]   #  这里将选择出的字符进行拼接
                    image = ImageCaptcha()
                    image.write(captcha_text,CaptchaImgPath + captcha_text + '.jpg')
                    count = count + 1
                    #sys.stdout.write("正在生成%d个验证码"%count)
                    sys.stdout.flush
                    
#  从生成的验证码中选取一部分用来后期的测试   
                
def prepare_test_set(CapchaImgPath,TestImgPath):
    
    fileNameList=[]
    for filepath in os.listdir(CapchaImgPath):
        capacha_image = filepath.split('/')[-1]
        fileNameList.append(capacha_image)
        
    random.seed(time.time())
    #  打乱验证码的排序规则
    random.shuffle(fileNameList)

    #   将选取的验证码图片放到测试文件中
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.move(CapchaImgPath + name, TestImgPath + name)
               
if __name__ == "__main__":
    
    generate_captcha_image(CHAR_SET,CHAR_SET_LEN,CAPTCHA_IMAGE_PATH)
    
    prepare_test_set(CAPTCHA_IMAGE_PATH,TEST_IMAGE_PATH)
    
    sys.stdout.write("Finish!")
    sys.stdout.flush
    

