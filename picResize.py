from PIL import Image
import os

def Resize(img):
    out = img.resize((227, 227))  # resize成299*299像素大小。
    return out
