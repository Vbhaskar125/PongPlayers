import numpy as np
from PIL import Image



def preprocess(image1,image2,image3,image4):
     Imgarray=[image1,image2,image3,image4]
     processedImg=[]
     for x in Imgarray:
         Colorimg = Image.fromarray(np.asarray(x), 'RGB')
         gray = Colorimg.convert("L")
         blackwhite = gray.point(lambda x: 0 if x < 128 else 255, '1')
         blackwhite = np.asarray(blackwhite)
         crop = blackwhite[32:198]
         processedImg.append(crop)

     one=Image.blend(Image.fromarray(processedImg[0], 'L'),Image.fromarray(processedImg[1], 'L'),0.5)
     two=Image.blend(Image.fromarray(processedImg[2], 'L'),Image.fromarray(processedImg[3], 'L'),0.5)
     blendedImg=Image.blend(one,two,0.5)
     return np.asarray(blendedImg)



