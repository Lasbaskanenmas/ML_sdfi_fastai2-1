import os
from PIL import Image
import numpy
import pandas as pd
import argparse
import shutil
import pathlib
from pathlib import Path



usage_example="example usage: \n "+r"TO DO"
# Initialize parser
parser = argparse.ArgumentParser(
                                 epilog=usage_example,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)



parser.add_argument("-f", "--Folder_path", help="path to dataset folder",required=True)
parser.add_argument("-a", "--All_filename", help="all.txt filename .eg 'esbjerg++.txt'",required=True)
parser.add_argument("-l", "--Labelsfoldername", help=".eg -l labelsByg/masks",required=True)



def main(folder_path,datatype,all_txt_filename,label_folder_name = "labels/masks",images_folder_name= "images"):
    """
    :param folder_path: path to dataset folder
    :param datatype: e.g '.tif'
    :param all_txt_filename: e.g 'esbjerg++.txt'
        
    :return: None
    """

    
    # 1. open a all.txt file that includes the paths to all images 
    # 2. replace all nan with 0 in the label images
    max_label=0
    min_label=0
    nr_of_labels={}
   
    with open(all_txt_filename, 'r') as f:
        all_list = f.readlines()
    for i in range(len(all_list)):
        print("i:"+str(i)+ " max label:"+str(max_label) +" min label:"+str(min_label),end='\r')#,end="\r",flush=True)
        path= folder_path+"/"+all_list[i].rstrip()
        #im= Image.open(path)
        #im.show()
        label_path = path.replace(images_folder_name,label_folder_name)
        #shutil.copy2(label_path,"tmp/"+Path(label_path).name)
        with Image.open(label_path) as label:
            numpy_label = numpy.array(label)
        #input(str(numpy_label))
        #set nan to 0
        where_are_NaNs = numpy.isnan(numpy_label)
        numpy_label[where_are_NaNs] =0
        #set everything that is larger than 1 to 0
        larger_than_one= numpy_label>1
        numpy_label[larger_than_one] = 0

        


        label_im = Image.fromarray(numpy_label)
        if numpy_label.flatten().max() > max_label:
            max_label = numpy_label.flatten().max()
        if numpy_label.flatten().min() < min_label:
            min_label = numpy_label.flatten().min()

        
        label_im.save("tmp/"+Path(label_path).name)




if __name__ == "__main__":
    args = parser.parse_args()
    main(folder_path=args.Folder_path,datatype=".tif", all_txt_filename=args.All_filename,label_folder_name = "labels/masks",images_folder_name= "images")


