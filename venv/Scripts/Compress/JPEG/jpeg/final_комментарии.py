#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
# для жсатия методом Хаффмана
import add.huffman as huffman
# встроенная библиотека для контейнеров, нужныж для сжатия
from collections import Counter
# для записи в файл
import json

class JPEG(object):
    def __init__(self):
        # С - матрица коэффициентов для ДКП
        self.C = np.full((8,8), 2/8)
        self.C[0,:] = np.sqrt(2/8*1/8)
        self.C[:,0] = np.sqrt(2/8*1/8)
        self.C[0,0] = 1/8
        
        # коэффициент для квантования
        self.Q = 20
        
    def DCT(self,img):
        # функция для ДКП
        
        # массив для хранения результата 
        img_uv = np.zeros( img.shape, dtype = np.float32)
        # коэффициенты
        y = np.arange(8)
        x = np.arange(8).reshape(-1,1)
       
        for u in range(8):
            for v in range(8):            
                cos_u = np.cos(u*np.pi*(2*x+1)/16)
                cos_v = np.cos(v*np.pi*(2*y+1)/16)
                
                img_uv[u,v] = np.sum(self.C[u,v] * img * cos_u * cos_v)
    
        return img_uv

#optimized version
    def IDCT(self,img):
        # функция для обратного ДКП
        img_uv = np.zeros( img.shape, dtype = np.float32)
        v = np.arange(8)
        u = np.arange(8).reshape(-1,1)
       
        for x in range(8):
            for y in range(8):            
                cos_u = np.cos(u*np.pi*(2*x+1)/16)
                cos_v = np.cos(v*np.pi*(2*y+1)/16)
                
                img_uv[x,y] = np.sum(self.C * img * cos_u * cos_v)
    
        return img_uv

    def quantification(self,img,Q):
        # квантификация
        v = np.arange(8)
        u = np.arange(8).reshape(-1,1)
        q = (u+v)*self.Q+1
        # np.fix - округление
        F = np.fix( img/q)
        return F


    def DEquantification(self, img,Q):
        # обратная квантификация
        v = np.arange(8)
        u = np.arange(8).reshape(-1,1)
        q = (u+v)*self.Q+1  
        F =  img * q
        
        return F

    def zigzag(self,img):
        # разворачивание массива в строку методом зигзага
        zig = []
        for i in range(0,8):
            if i % 2 == 1:
                zig.extend( np.diag(np.fliplr(img[:i+1,:i+1])) ) 
            else:
                zig.extend(np.diag(np.fliplr(img[:i+1,:i+1]))[::-1]) 
                
        for i in range(1,8):
            if i % 2 == 0:
                zig.extend( np.diag(np.fliplr(img[ i:,i: ])) ) 
            else:
                zig.extend(np.diag(np.fliplr(img[ i:,i: ]))[::-1])         
        return zig


    def DEzigzag(self,zig):
        # сворачивание строки  в массив методом обратным методу зигзага
        img = np.zeros((8,8))
           
        bl2tr = True
        
        for i in range(0,8):
            if bl2tr:
                for j in range(0,i+1):
                    img[i-j,j] = zig.pop(0)
                bl2tr = False
            else:
                for j in range(0,i+1):
                    img[j,i-j] = zig.pop(0)
                bl2tr = True
           
        img = np.rot90(img,2)
        bl2tr = True
        for i in range(0,7):
            if bl2tr:
                for j in range(0,i+1):
                    img[i-j,j] = zig.pop(-1)
                bl2tr = False
            else:
                for j in range(0,i+1):
                    img[j,i-j] = zig.pop(-1)
                bl2tr = True   
        
        img = np.rot90(img,2)   
        return img

    def RLE(self,vector):
        # кодирование серий
        k_0 = 0
        rle = []
        vector.append('eof')
        for i in vector:
            if i == 0:
                k_0+=1
            else:
                rle.append((k_0,i))
                k_0 = 0 
                
        return rle

    def DE_RLE(self,rle):
        # декодирование серий
        zig = []
        for (i,j) in rle:
            if j =='eof':
                zig.extend([0]*i)
            else:
                zig.extend([0]*i)
                zig.append(j)
        
        return zig
            
    def coding(self,_rle):
        # кодирование методом Хаффмана
        count = Counter(_rle)
        dictionary=huffman.codebook(count.items())
    
        code = []
        for i in _rle:
            code.append( dictionary[i])
        
        return code, dictionary
    
    def get_key_from_value(self,dictOfElements, valueToFind):
        # получение ключа словаря по значению
        listOfKeys = list()
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
        return  listOfKeys
    
    def decoding(self,code, dictionary):
        # декодирование методом Хаффмана
        
        rle = []
        
        for i in code:
            t = self.get_key_from_value(dictionary,i)
            rle.extend(t)
            
        return rle

    def jpeg(self,window, Q = 20):
        # функция, которая для окна 8х8 делаёт сжатие JPEG
        
        # cv2.dct - встроенная функция для DCT, работает быстрее
        # DCTed = cv2.dct(window)
        DCTed = self.DCT(window)
        
        # квантификация
        quanted = self.quantification(DCTed, self.Q)
        # зиигзагирование
        zigzaged = self.zigzag(quanted)
        # кодирование серий
        rle = self.RLE(zigzaged)
        # кодирование Хаффмана
        coded, dicted = self.coding(rle)
        
        return coded, dicted
    
    def de_jpeg(self,coded, dicted,Q = 20):
        # функция, которая для кода хаффмана восстанавливает изображение
        
        # декодирование Хаффмана
        decoded = self.decoding(coded, dicted)
        
        # декодирование серий
        derle = self.DE_RLE(decoded)
        
        # из строки получается исходный массив; обратный зигзаг
        dezigzaged = self.DEzigzag(derle)
        
        # деквантификация
        dequanted = self.DEquantification(dezigzaged,self.Q)
        
        # обратное дкп, cv2.idct - встроенная функция для этого, работает быстрее
        # deDCTed = cv2.idct(dequanted)
        deDCTed = self.IDCT(dequanted)
        
        return deDCTed
    
    def encode(self,image):
        # функция для сжатия, использует вышереализованные функции для сжатия
        
        # создаём текстовый файл
        file = open('image.ourjpg', 'w+')
        
        # переводим изображение в цветовое простарнство YCrCb
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb )
        ycrcb_image = ycrcb_image.astype(np.float32)


        # цикл - пробегаем все 8х8 участки изображения, p - отвечает за каналы
        for i in range(0,64):
            for j in range(0,64):
                for p in range(0,3):
                    
                    # берём окно 8х8. m:n это срез, то есть элементы с m позиции
                    # по n-1 позицию
                    window = ycrcb_image[ 8*i:8*i+8 ,8*j:8*j+8 , p]
                    
                    # получаем код Хаффмана для этого окна
                    a,b = self.jpeg(window, self.Q)
                    
                    # переводим список в строку для записи
                    a = json.dumps(a)
                    
                    # переводим словарь в строку для записи 
                    b = { json.dumps(i) : j for (i,j) in b.items()}
                    b = json.dumps(b)

                    # записываем полученные строки
                    file.write(a)
                    file.write('\r\n')
                    file.write(b)
                    file.write('\r\n')
                    
    def decode(self, path='image.ourjpg'):
        # функция для декодирования
        file = open(path, 'r')
        
        # массив для результата
        result = np.zeros((512,512,3), dtype=np.uint8)
        
        for i in range(0,64):
            for j in range(0,64):
                for p in range(0,3):
                    
                    # читаем файл
                    a = file.readline()
                    _ = file.readline()
                    b = file.readline()
                    _ = file.readline()
                    
                    # раскодируем строки
                    a = json.loads(a)
                    b = json.loads(b)
                    
                    b = { tuple(json.loads(i)):j for (i,j) in b.items()  }
                       
                    # для считанных кодов восстанавливаем изображение (с некоторыми потреями)
                    result[8*i:8*i+8 ,8*j:8*j+8 , p] = self.de_jpeg(a,b,self.Q)
        
        return result
    
image = cv2.imread('lena.bmp')
image = cv2.resize(image, (512,512))

decoder = JPEG()

decoder.encode(image)
#%%

res = decoder.decode()

res = cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)    
    
cv2.imshow('source_image', image)

cv2.imshow('our_jpeg', res)

cv2.waitKey()
cv2.destroyAllWindows()
