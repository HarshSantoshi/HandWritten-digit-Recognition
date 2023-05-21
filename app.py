import pygame,sys
from pygame.locals import *
import cv2
import numpy as np
from keras.models import load_model
import cv2
#BY HARSH SANTOSHI (2k21/SE/82) AND HARSH(2K21/SE/77)
WINDOWSIZEX = 640
WINDOWSIZEY = 480
saveImage = False
MY_MODEL = load_model("MyModel.h5")
LABELS = {0:"ZERO" , 1:"ONE" , 2:"TWO" , 3:"THREE" , 4:"FOUR" , 5: "FIVE" , 6:"SIX", 7:"SEVEN", 8:"EIGHT",      9:"NINE", 10:"TEN"}
pygame.init()
FONT = pygame.font.Font('freesansbold.ttf',20)
SURFACE= pygame.display.set_mode((640, 480))
pygame.display.set_caption("Digit Prediction")
writing=False
predict = True
imgCount=1
x_coord_arr = []
y_coord_arr = []
while True:
    for event in pygame.event.get():
        #To close the window
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and writing:
            x_coord , y_coord = event.pos
            pygame.draw.circle(SURFACE , (255,255,255), (x_coord , y_coord) ,  3,  0)
            x_coord_arr.append(x_coord)
            y_coord_arr.append(y_coord)
        if event.type == MOUSEBUTTONDOWN:
            writing=True
        if event.type == MOUSEBUTTONUP:
            writing=False
            x_coord_arr=sorted(x_coord_arr)
            y_coord_arr=sorted(y_coord_arr)    
            #Creating a boundary 
            sq_min_x ,sq_max_x = max(x_coord_arr[0]- 5,0) , min(640,x_coord_arr[-1]+5)
            sq_min_y ,sq_max_y = max(y_coord_arr[0]- 5,0) , min(480,y_coord_arr[-1]+5)
            x_coord_arr =[]
            y_coord_arr =[]
            arr_image = np.array(pygame.PixelArray(SURFACE))[sq_min_x :sq_max_x , sq_min_y:sq_max_y].T.astype(np.float32)
            if predict:
                img =cv2.resize(arr_image , (28,28))
                img = np.pad(img , (10,10) , 'constant' , constant_values = 0)
                img  = cv2.resize(img , (28,28))/255
                target = str(LABELS[np.argmax(MY_MODEL.predict(img.reshape(1,28,28,1)))])
                OutputText = FONT.render(target , True , (0,0,255), (255,255,255))
                OutputRect = OutputText.get_rect()
                OutputRect.left , OutputRect.bottom = sq_min_x, sq_min_y
                SURFACE.blit(OutputText, OutputRect)    
    pygame.display.update()
