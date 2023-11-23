from PIL import Image
import os
import numpy as np
import pandas as pd
import TongueDetect
import TrainSVM
import time
import picResize

#EXTRACT THE DATA AND LABEL FROM TONGUE IMAGES
def analysis(img):
    img=picResize.Resize(img)
    clf=TrainSVM.trainSVM()  
    colorPixels=img.convert("RGB")
    colorArray=np.array(colorPixels.getdata()).reshape(img.size+(3,))
    indicesArray=np.moveaxis(np.indices(img.size),0,2)
    #reshape the array 
    allArray=np.dstack((indicesArray,colorArray)).reshape((-1,5))
    df=pd.DataFrame(allArray,columns=["col","row","red","green","blue"])
    #Label each dataframe with the file name
    coat,body=TongueDetect.detection(df,clf)
    return body,coat
    #print("舌质和舌苔的颜色分别是",body,coat)



   

