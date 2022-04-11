import os
from zipfile import ZipFile
import cv2
import np as np
from PIL import Image

if __name__ == '__main__':

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile('/Users/cody/Downloads/lesion_images.zip', 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for fileName in listOfFileNames:
            # Check filename endswith csv
            if fileName.endswith('.png'):

                tmp_file = 'original.png'
                # CV2
                original_file = open(tmp_file, "wb")
                zipObj.read(fileName)
                original_file.write(zipObj.read(fileName))
                original_file.close()

                img_np = cv2.imread(tmp_file, -1)
                print(type(img_np))

                #nparr = np.frombuffer(zipObj.read(fileName), dtype=np.int16)
                #img_np = cv2.imread(nparr, -1)

                #print(type(nparr))
                #img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
                #print(type(img_np))
                img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
                #img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))

                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                #img_np = clahe.apply(img_np)


                #img_np = cv2.equalizeHist(img_np)

                #img_np = (img_np.astype(np.int32) - 32768).astype(np.int16)

                im = Image.fromarray(img_np)

                #imgGray = im.convert('L')
                #imgGray.save('test_gray.jpg')

                im.save("filename_0.png")

                #im.save("filename_1.jpeg")

                #print(type(zipObj.read(fileName)))
                #print(fileName)
                # Extract a single file from zip
                base_file = os.path.basename(fileName)
                #print(base_file)
                #zipObj.extract(fileName, 't')
                exit(0)

