import argparse
import csv
import hashlib
import os
import shutil
from zipfile import ZipFile
import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd

md5_checksums = dict()
coords_df = pd.read_csv('DL_info.csv')

def get_md5():
    with open("MD5_checksums.txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            row_s = line.strip().split("  ")
            check = row_s[0]
            file = row_s[1]
            md5_checksums[file] = check


def convert_to_png(ims):
    """save 2D slices to 3D nifti file considering the spacing"""
    if len(ims) < 300:  # cv2.merge does not support too many channels
        V = cv2.merge(ims)
    else:
        V = np.empty((ims[0].shape[0], ims[0].shape[1], len(ims)))
        for i in range(len(ims)):
            V[:, :, i] = ims[i]

    return V


def verify_md5(file_path):
    filename = os.path.basename(file_path)
    check = md5_checksums[filename]

    if check is not None:

        calc = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        if calc == check:
            print(file_path + ": Checksum OK (" + calc + ":" + check + ")")
            return True
        else:
            print(file_path + ": Checksum NOT OK (" + calc + ":" + check + ")")
    else:
        print(file_path + ": No md5 found!")

    return False


def checkzips(args):
    checked_zips = []

    # check that file exists
    allFiles = True
    # get list of source zips
    onlyfiles = [f for f in listdir(args.src_zip_path) if isfile(join(args.src_zip_path, f))]

    for md5_file, md5 in md5_checksums.items():
        if md5_file not in onlyfiles:
            allFiles = False
            print('Missing zip: ' + md5_file)

    # check MD5 hash on file
    allMD5 = True
    if allFiles:

        for file in onlyfiles:
            file_path = os.path.join(args.src_zip_path, file)
            #if not verify_md5(file_path):
            #    allMD5 = False
            #    print('Bad MD5: ' + file)
            #else:
            checked_zips.append(file_path)
    else:
        print('Not all zips found')

    if allMD5:
        return checked_zips

    else:
        print('Not all zips passed MD5 check')

    return checked_zips.clear()


# border box coord converter
def convert(x1, y1, x2, y2, image_width, image_height):  # may need to normalize
    dw = 1. / image_width
    dh = 1. / image_height
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def get_coords(coords):

    xmin = float(coords[0])
    ymin = float(coords[1])
    xmax = float(coords[2])
    ymax = float(coords[3])

    w_img = 512
    h_img = 512

    w = xmax - xmin
    h = ymax - ymin

    xcenter = (xmin + w / 2) / w_img
    ycenter = (ymin + h / 2) / h_img
    w = w / w_img
    h = h / h_img

    return xcenter, ycenter, w, h


def read_DL_info():
    """read spacings and image indices in DeepLesion"""
    spacings = []
    idxs = []
    with open('DL_info.csb', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rownum = 0
        for row in reader:
            if rownum == 0:
                header = row
                rownum += 1
            else:
                idxs.append([int(d) for d in row[1:4]])
                spacings.append([float(d) for d in row[12].split(',')])

    idxs = np.array(idxs)
    spacings = np.array(spacings)
    return idxs, spacings

def save_file(zipObj, file_path, zip_file_path):

    original_file = open(file_path, "wb")
    original_file.write(zipObj.read(zip_file_path))
    original_file.close()

def convert_save_file(zipObj, file_path, zip_file_path):


    tmp_file = 'tmpfile.png'
    original_file = open(tmp_file, "wb")
    original_file.write(zipObj.read(zip_file_path))
    original_file.close()

    #img_np = np.frombuffer(zipObj.read(zip_file_path), dtype=np.int16)
    #img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_COLOR in OpenCV 3.1
    img_np = cv2.imread(tmp_file, -1)
    img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # bgr to gray
    im = Image.fromarray(img_np)
    im.save(file_path)

    '''
    tmp_file = 'tmpfile.png'
    original_file = open(tmp_file, "wb")
    original_file.write(zipObj.read(zip_file_path))
    original_file.close()

    img_np = cv2.imread(tmp_file, -1)
    img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))
    im = Image.fromarray(img_np)
    im.save(file_path)
    '''

    os.remove(tmp_file)

def get_zip_image_name(zip_image_path):

    splt_path = zip_image_path.split("/")
    image_name = splt_path[1] + '_' + splt_path[2]
    return image_name

def build_dataset(args, zip_file_path):

    cord_file_list = coords_df["File_name"].tolist()

    with ZipFile(zip_file_path, 'r') as zipObj:
        # Get a list of all archived file names from the zip
        print('Opening Zip: ' + zip_file_path)
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for image_path in listOfFileNames:
            # Check filename endswith csv
            if image_path.endswith('.png'):

                image_name = get_zip_image_name(image_path)

                if image_name in cord_file_list:
                    name = coords_df.loc[coords_df['File_name'].str.contains(image_name, case=False)]

                    coords = coords_df["Bounding_boxes"][name.index[0]].split(',')
                    #ncoords = coords_df["Normalized_lesion_location"][name.index[0]].split(',')

                    #label = coords_df["Coarse_lesion_type"][name.index[0]]
                    #everything is a lesion
                    label = 1
                    train_val_test = coords_df["Train_Val_Test"][name.index[0]]
                    '''
                    x, y, w, h = convert(float(coords[0]), float(coords[1]), float(coords[2]),
                                         float(coords[3]),
                                         512, 512)

                    if (x > 1) or (y > 1) or (w > 1) or (h > 1):
                        print('file: ' + image_name)
                        print('original: ' + str(coords))
                        print('sam: ' + str(
                            convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)))
                        print('cody: ' + str(get_coords(coords)))
                        print('normal original: ' + str(ncoords))
                        exit(0)
                    '''
                    '''
                    img_np = cv2.imread(tmp_file, -1)
                    img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))
                    im = Image.fromarray(img_np)
                    im.save("filename_0.png")
                    '''

                    # Train
                    if train_val_test == 1:
                        train_image_path = os.path.join(args.dst_data_path, 'train', 'images', image_name)
                        #save_file(zipObj, train_image_path, image_path)
                        convert_save_file(zipObj, train_image_path, image_path)


                        train_label_path = os.path.join(args.dst_data_path, 'train', 'labels', image_name[:-3] + "txt")

                        with open(train_label_path, 'a') as f:
                            x, y, w, h = convert(float(coords[0]), float(coords[1]), float(coords[2]),
                                                 float(coords[3]),
                                                 512, 512)
                            f.write(str(label - 1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

                        print('Training image: ' + image_name + ' from: ' + zip_file_path + ' ' + str(train_val_test))

                    # Validation
                    elif train_val_test == 2:

                        val_image_path = os.path.join(args.dst_data_path, 'val', 'images', image_name)
                        #save_file(zipObj, val_image_path, image_path)
                        convert_save_file(zipObj, val_image_path, image_path)

                        val_label_path = os.path.join(args.dst_data_path, 'val', 'labels', image_name[:-3] + "txt")
                        with open(val_label_path, 'a') as f:
                            x, y, w, h = convert(float(coords[0]), float(coords[1]), float(coords[2]),
                                                 float(coords[3]),
                                                 512, 512)
                            f.write(str(label - 1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                        print('Val image: ' + image_name + ' from: ' + zip_file_path + ' ' + str(train_val_test))

                    # Test
                    elif train_val_test == 3:

                        test_image_path = os.path.join(args.dst_data_path, 'test', 'images', image_name)
                        #ave_file(zipObj, test_image_path, image_path)
                        convert_save_file(zipObj, test_image_path, image_path)

                        test_label_path = os.path.join(args.dst_data_path, 'test', 'labels', image_name[:-3] + "txt")
                        with open(test_label_path, 'a') as f:
                            x, y, w, h = convert(float(coords[0]), float(coords[1]), float(coords[2]),
                                                 float(coords[3]),
                                                 512, 512)
                            f.write(str(label - 1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                        print('Test image: ' + image_name + ' from: ' + zip_file_path + ' ' + str(train_val_test))


    zipObj.close()


def process_zip():
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile('src_zips/Images_png_56.zip', 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        for fileName in listOfFileNames:
            # Check filename endswith csv
            if fileName.endswith('.png'):
                tmp_file = 'original.png'
                # CV2
                #original_file = open(tmp_file, "wb")
                #zipObj.read(fileName)
                #original_file.write(zipObj.read(fileName))
                #original_file.close()

                #img_np = cv2.imread(tmp_file, -1)
                #img_np = cv2.imread(zipObj.read(fileName))




                print(fileName)
                convert_save_file(zipObj, 'test.png', fileName)
                '''
                fileName = 'Images_png/004408_01_02/088.png'
                print(fileName)
                convert_save_file(zipObj, 'test2.png', fileName)
                '''

                '''
                nparr = np.frombuffer(zipObj.read(fileName), dtype=np.int16)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
                img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))
                im = Image.fromarray(img_np)
                im.save("filename_0.png")
                '''

                #img_np = cv2.imread(tmp_file, -1)
                #print(type(img_np))

                #nparr = np.frombuffer(zipObj.read(fileName), dtype=np.int16)
                # img_np = cv2.imread(nparr, -1)

                # print(type(nparr))
                #img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
                # print(type(img_np))

                #img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
                # img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))

                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # img_np = clahe.apply(img_np)

                # img_np = cv2.equalizeHist(img_np)

                #img_np = (img_np.astype(np.int32) - 32768).astype(np.int16)

                #im = Image.fromarray(img_np)

                # imgGray = im.convert('L')
                # imgGray.save('test_gray.jpg')

                #im.save("filename_0.png")

                #im.save("filename_1.jpeg")

                # print(type(zipObj.read(fileName)))
                # print(fileName)
                # Extract a single file from zip
                base_file = os.path.basename(fileName)
                # print(base_file)
                # zipObj.extract(fileName, 't')
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Lesion PreProcessor')

    # general args
    parser.add_argument('--src_zip_path', type=str, default='src_zips', help='location of zip files')
    parser.add_argument('--dst_data_path', type=str, default='lesion_data', help='location of extracted images')

    # get args
    args = parser.parse_args()

    #process_zip()
    #exit(0)

    # get md5 list
    get_md5()

    # check that the dataset is valid
    valid_zips = checkzips(args)
    #valid_zips = []
    #valid_zips.append('src_zips/Images_png_56.zip')

    # make directories

    if os.path.exists(args.dst_data_path):
        shutil.rmtree(args.dst_data_path)

    # Create a new directory because it does not exist
    os.makedirs(os.path.join(args.dst_data_path, 'train', 'images', ))
    os.makedirs(os.path.join(args.dst_data_path, 'val', 'images', ))
    os.makedirs(os.path.join(args.dst_data_path, 'test', 'images', ))
    os.makedirs(os.path.join(args.dst_data_path, 'train', 'labels', ))
    os.makedirs(os.path.join(args.dst_data_path, 'val', 'labels', ))
    os.makedirs(os.path.join(args.dst_data_path, 'test', 'labels', ))
    print("New dataset director created")


    for valid_zip in valid_zips:
        build_dataset(args, valid_zip)


