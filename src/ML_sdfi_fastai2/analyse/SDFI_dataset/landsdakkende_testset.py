
#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import shutil
import os
import argparse
def get_fixed_set(prediction_base_path,labels_base_path,images_base_path):
    """
    @arg prediction_base_path: path to the location where the predictions are saved. e.g : path/to/models/experiment/modelname/
    @return a dictionary of type {"images":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"labels":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"predictions":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']}}
    """

    prediction_base_path= Path(prediction_base_path)



    rosenborg=Path("2021_84_40_08_0619_1000_3000.tif")
    borgen = Path("2021_84_40_08_0619_11000_1000.tif")
    rundetorn = Path("2021_84_40_08_0619_6000_4000.tif")
    torvehallerne = Path("2021_84_40_08_0619_3000_8000.tif")
    noma=Path("2021_84_40_11_0702_10000_22000.tif")
    kanonhallarna=Path("2021_84_40_11_0702_12000_24000.tif")
    amagerbakke=[Path("2021_84_40_11_0702_6000_16000.tif"),Path("2021_84_40_11_0703_0_16000.tif")]
    vind_molle= Path("2021_84_40_11_0703_14000_4000.tif")
    lufthavn= Path("2021_84_40_12_0736_4000_15000.tif")
    kende_bygninger= [rosenborg,borgen,rundetorn,torvehallerne,noma,kanonhallarna,vind_molle,lufthavn]+amagerbakke
    industri_skov = Path("2021_84_40_04_0289_11000_16000.tif")
    industri = [Path("2021_84_40_04_0288_16000_19000.tif"),Path("2021_84_40_04_0289_2000_5000.tif"),Path("2021_84_40_04_0289_1000_6000.tif"),industri_skov,Path("2021_82_24_05_0418_1000_19000.tif")] #removed by peter ,Path("2021_84_40_11_0702_5000_14000.tif")
    #this image have bad label  smabadshavn = Path("2021_84_40_04_0289_2000_1000.tif")
    roundabout=Path("2021_84_40_04_0289_10000_22000.tif")
    vej_skov_vandlob= Path("2021_84_40_04_0289_11000_11000.tif")
    veje = [roundabout,vej_skov_vandlob]
    skov_vand_forstad =Path("2021_84_40_04_0289_14000_20000.tif")
    kolonihaver =Path("2021_84_40_11_0702_16000_18000.tif")
    kyst_skov=Path("2021_85_44_003_0332_12000_22000.tif")
    kyst =[Path("2021_85_44_003_0332_11000_17000.tif"),kyst_skov]
    sma_soer= Path("2021_85_44_003_0332_12000_13000.tif")
    juletraer=Path("2021_85_46_02_0125_0_12000.tif")
    skov = [Path("2021_85_46_02_0125_0_17000.tif"),juletraer]
    skov_mark=Path("2021_82_21_01_0054_10000_0.tif")
    land_mark_huse= Path("2021_85_46_02_0125_1000_7000.tif")
    land_vandlob=Path("2021_85_46_02_0125_4000_2000.tif")
    land_vej_vandlob_traer=Path("2021_85_46_02_0125_4000_16000.tif")
    land=[Path("2021_85_46_02_0125_5000_9000.tif"),Path("2021_85_48_17_0503_12000_2000.tif"),land_mark_huse,skov_mark]+skov
    oreglered_vandlob = Path("2021_85_46_02_0126_16000_7000.tif")
    forstad_jernebane =  Path("2021_85_47_07_0273_12000_2000.tif")
    so=Path("2021_85_48_16_0455_3000_9000.tif")
    storhavn=Path("2021_83_25_11_0533_3000_2000.tif")
    havn=[Path("2021_82_24_05_0417_14000_19000.tif"),storhavn] # this image have bad label smabadshavn]
    vandlob=[Path("2021_82_24_05_0417_13000_3000.tif"),oreglered_vandlob,land_vej_vandlob_traer,land_vandlob,sma_soer,so,Path("2021_85_48_13_0373_1000_13000.tif"), Path("2021_85_48_13_0373_1000_16000.tif"),Path("2021_85_48_13_0373_10000_21000.tif"),Path("2021_84_40_04_0289_0_24000.tif")]
    bykerne=[Path("2021_82_24_05_0418_1000_10000.tif"),Path("2021_84_40_08_0618_7000_10000.tif"),Path("2021_82_24_05_0417_16000_14000.tif")]
    forstad =  [Path("2021_84_40_04_0289_14000_12000.tif"),skov_vand_forstad,kolonihaver,Path("2021_82_24_04_0394_9000_22000.tif"),forstad_jernebane]
    image_with_ignore1 = Path("2021_85_48_17_0504_14000_23000.tif")
    images_with_ignore=[image_with_ignore1]


    image_paths_for_subsets= {}
    label_paths_for_subsets = {}
    prediction_paths_for_subsets = {}

    #images
    #image_paths_for_subsets["images_with_ignore"]= [images_base_path/im for im in images_with_ignore]
    image_paths_for_subsets["SUBURB"]= [images_base_path/im for im in forstad]
    image_paths_for_subsets["URBAN"]= [images_base_path/im for im in bykerne]
    image_paths_for_subsets["RURAL"] = [images_base_path / im for im in land]
    image_paths_for_subsets["INDUSTRY"] = [images_base_path / im for im in industri]
    image_paths_for_subsets["LAKES AND STREAMS"] = [images_base_path / im for im in vandlob]
    image_paths_for_subsets["COASTS"] = [images_base_path / im for im in kyst]
    image_paths_for_subsets["PORTS"] = [images_base_path / im for im in havn]
    image_paths_for_subsets["ROADS"] = [images_base_path / im for im in veje]
    image_paths_for_subsets["WELL KNOWN PLACES"]= [images_base_path/im for im in kende_bygninger]

    #labels
    #label_paths_for_subsets["images_with_ignore"]= [labels_base_path/im for im in images_with_ignore]
    label_paths_for_subsets["SUBURB"]= [labels_base_path/im for im in forstad]
    label_paths_for_subsets["URBAN"]= [labels_base_path/im for im in bykerne]
    label_paths_for_subsets["RURAL"] = [labels_base_path / im for im in land]
    label_paths_for_subsets["INDUSTRY"] = [labels_base_path / im for im in industri]
    label_paths_for_subsets["LAKES AND STREAMS"] = [labels_base_path / im for im in vandlob]
    label_paths_for_subsets["COASTS"] = [labels_base_path / im for im in kyst]
    label_paths_for_subsets["PORTS"] = [labels_base_path / im for im in havn]
    label_paths_for_subsets["ROADS"] = [labels_base_path / im for im in veje]
    label_paths_for_subsets["WELL KNOWN PLACES"]= [labels_base_path/im for im in kende_bygninger]

    #SUBURB, URBAN, RURAL, INDUSTRY, LAKES AND STREAMS, COASTS, PORTS, ROADS, WELL KNOWN PLACES
    #predictions
    #prediction_paths_for_subsets["images_with_ignore"]= [prediction_base_path/im for im in images_with_ignore]
    prediction_paths_for_subsets["SUBURB"]= [prediction_base_path/im for im in forstad]
    prediction_paths_for_subsets["URBAN"]= [prediction_base_path/im for im in bykerne]
    prediction_paths_for_subsets["RURAL"] = [prediction_base_path / im for im in land]
    prediction_paths_for_subsets["INDUSTRY"] = [prediction_base_path / im for im in industri]
    prediction_paths_for_subsets["LAKES AND STREAMS"] = [prediction_base_path / im for im in vandlob]
    prediction_paths_for_subsets["COASTS"] = [prediction_base_path / im for im in kyst]
    prediction_paths_for_subsets["PORTS"] = [prediction_base_path / im for im in havn]
    prediction_paths_for_subsets["ROADS"] = [prediction_base_path / im for im in veje]
    prediction_paths_for_subsets["WELL KNOWN PLACES"]= [prediction_base_path/im for im in kende_bygninger]


    result={"images":image_paths_for_subsets,"labels":label_paths_for_subsets,"predictions":prediction_paths_for_subsets}
    return result

if __name__ == "__main__":
    #testing the functionality by printing out the dictionary and make a folder containing all info related to the fixed set of images
    parser = argparse.ArgumentParser()
    parser.add_argument("--Predictions_folder",help="e.g path\to\models\modelname\savedmodelfolder  , folder where predictions are located",
						required=True)
    parser.add_argument("--Copy",help="e.g True or False, copy files to this folder", required=True)
    args = parser.parse_args()


    dictionary=get_fixed_set(prediction_base_path=args.Predictions_folder)

    print("dictionary:"+str(dictionary))
    if str(args.Copy) == "True":
        for image_type in dictionary.keys():
            #image_type=e.g images labels predictions
            for subset in dictionary[image_type]:
                #subset =e.g sommerhuse,land,bykerne
                for image_path in dictionary[image_type][subset]:
                    folder= "fixed_set_predictions/"+image_type+"/"+subset
                    os.makedirs(folder,exist_ok = True)
                    #copy images
                    shutil.copy(image_path, folder+"/"+image_path.name)
