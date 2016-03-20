from PIL import Image, ImageOps
from numpy import asarray, ravel

def read():
    digits = []
    for d in range(10):
        fn = "digits/"+str(d)+".png"
        img  = ImageOps.invert(Image.open(fn).convert('L'))
        digits.append(ravel(asarray(img)))
    return digits