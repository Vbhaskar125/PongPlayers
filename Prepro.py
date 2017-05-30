import numpy as np
from PIL import Image
import skimage

class Preprocesssor():
    def prepro(self,image1,image2,image3,image4):
         Imgarray=[image1,image2,image3,image4]
         for x in Imgarray:
             img = Image.fromarray(np.asarray(x), 'RGB')
             gray = img.convert("L")
             bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
             bw = np.asarray(bw)
             crop = bw[32:198]
             crop=Image.fromarray(crop)
             crop.show()


