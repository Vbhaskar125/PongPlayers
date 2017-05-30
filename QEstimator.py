from PIL import Image
import numpy as np

image1=Image.open('im3g.JPG')
image2=Image.open('im4g.JPG')
image3=Image.open('im5g.JPG')
image4=Image.open('im6g.JPG')

Imgarray=[image1,image2,image3,image4]
finalImage=0
for x in Imgarray:
 img = Image.fromarray(np.asarray(x), 'RGB')
 #img.show()
 gray = img.convert("L")
 bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
 bw = np.asarray(bw)
 crop = bw[32:198]
 finalImage +=crop
 crop=Image.fromarray(crop,'L')
 #crop.show()

zxp=Image.fromarray(finalImage,'L')
zxp.show()
