import os
import argparse
import pathlib

# Initialize parser
parser = argparse.ArgumentParser()


parser.add_argument("-v", "--Validtxtfile", help="path to .txt file that includes the validation images",required=True)
parser.add_argument("-a", "--Alltxtfile", help="path to .txt file that includes all images for training and validation",required=True)
parser.add_argument("-p", "--Prefix", help="prefix for the train.txt file  . e.g  test ->  test_train.txt",required=False)

def create_train_txt(path_to_all_txt,path_to_valid_txt,name_prefix=""):
    """
    create a train.txt file based on all.txt and valid.txt
    @returns: path_to_created_train_txt_file
    """

    path_to_all_txt= pathlib.Path(path_to_all_txt)
    path_to_valid_txt = pathlib.Path(path_to_valid_txt)
    path_to_train_txt = path_to_all_txt.parents[0]/ (name_prefix+"train.txt")
    train_lines=[] #all files not present in valid.txt

    with open(path_to_all_txt,"r") as all_file:
        all_lines =  all_file.readlines()
    with open(path_to_valid_txt,"r") as valid_file:
        valid_lines =  valid_file.readlines()
    for line in all_lines:
        found_file = False
        for valid_line in valid_lines:
            if valid_line in line:
                found_file = True
                break
        if not found_file:
            train_lines.append(line)
    #write the lines to train.txt
    with open(path_to_train_txt,"w") as train_file:
        train_file.write("".join(train_lines))
    return path_to_train_txt

if __name__ == "__main__":
    args = parser.parse_args()

    if args.Prefix:
        train_trefix = args.Prefix
    else:
        train_trefix =""


    create_train_txt(path_to_all_txt=args.Alltxtfile,path_to_valid_txt=args.Validtxtfile,name_prefix=train_trefix)
