
import pathlib
import image_from_mask
import random

if __name__ == "__main__":
    example_usage= r"python visualize_all_images_in_text_file.py -p T:\mnt\logs_and_models\befastelse\befastelse_iteration_06\models\iteration_6 -i T:\mnt\trainingdata\befastelse\iteration_6\images -l T:\mnt\trainingdata\befastelse\iteration_6\labels\masks -s -t T:\mnt\trainingdata\befastelse\iteration_6\valid.txt"
    print("########################EXAMPLE USAGE########################")
    print(example_usage)
    print("#############################################################")
    import argparse

    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--Textfile", help="path/to/textfile.txt  e.g valid.txt",required=True)
    parser.add_argument("-i", "--ImageFolder", help="path/to/folder  e.g path/to/images",required=True)
    parser.add_argument("-l", "--LabelFolder", help="path/to/folder  e.g path/to/masks",required=True)
    parser.add_argument("-p", "--PredictionFolder", help="path/to/folder  e.g path/to/predictionfolder",required=True)
    parser.add_argument("-a", "--AfterBurnFolder", help="path/to/folder  e.g path/to/knownhouses",required=False)


    parser.add_argument('-s','--Save',
                        help='should we save the image to disk -s-> True , no -s -> False: ',action = 'store_true',default=False)
    parser.add_argument('-v','--Show',
                        help='should we show images to screen? -s-> True , no -s -> False: ',action = 'store_true',default=False)
    parser.add_argument('-o','--Overlay',
                        help='should we oshow the visualization ontop of the original image? -o-> True , no -o -> False: ',action = 'store_true',default=False)
    parser.add_argument('-r','--Randomize_order',
                        help='-r-> True , no -r -> False: ',action = 'store_true',default=False)



    args = parser.parse_args()
    with open(args.Textfile) as f:
        lines= f.readlines()
        if args.Randomize_order:
            random.shuffle(lines)
        for line in lines:
            print(line)
            name = pathlib.Path(line.rstrip())
            if args.AfterBurnFolder:
                afterburn_path=pathlib.Path(args.AfterBurnFolder)/name
            else:
                afterburn_path = None


            if args.Overlay:
                image_from_mask.masked_image_from_image_prediction_label(pathlib.Path(args.ImageFolder)/name,pathlib.Path(args.LabelFolder)/name,pathlib.Path(args.PredictionFolder)/name,save = args.Save,show= args.Show,class_remappings={255:0},afterburn_path=afterburn_path)
            else:
                #mapp 255 to 0 in order for visualizations to be able to use a small range of colors
                image_from_mask.visualize_triplet(pathlib.Path(args.ImageFolder)/name,pathlib.Path(args.LabelFolder)/name,pathlib.Path(args.PredictionFolder)/name,save = args.Save,show= args.Show,class_remappings={255:0},afterburn_path=afterburn_path)
