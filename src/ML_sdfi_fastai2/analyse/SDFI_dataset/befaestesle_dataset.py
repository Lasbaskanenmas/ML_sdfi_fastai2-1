from pathlib import Path
def get_fixed_set(prediction_base_path,labels_base_path,images_base_path):
    """
    @arg prediction_base_path: path to the location where the predictions are saved. e.g : path/to/models/experiment/modelname/
    @return a dictionary of type {"images":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"labels":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']},"predictions":{"sommerhuse":['path/to/image.png'],"villahuse":['path/to/image.png'],"industri":['path/to/image.png']}}
    """

    prediction_base_path= Path(prediction_base_path)



    land=[Path("O2021_84_41_1_0003_00074400_3000_4000.tif"),Path("O2021_83_31_1_0042_00092126_8000_0.tif"),Path("O2021_83_31_1_0042_00092126_7000_1000.tif"),Path("O2021_83_31_1_0042_00092126_4000_4000.tif"),Path("O2021_83_31_1_0042_00092126_4000_0.tif"),Path("O2021_83_31_1_0042_00092126_0_0.tif"),Path("O2021_83_31_1_0042_00092126_3000_0.tif")] 
    bykerne=[Path("O2021_82_24_1_0021_00002042_10cm_0_0.tif"),Path("O2021_82_24_1_0021_00002042_10cm_0_2000.tif"),Path("O2021_82_24_1_0021_00002042_10cm_8000_0.tif")]
    paths=[Path("O2021_84_41_1_0003_00074400_8200_3000.tif")]
    forstad =  [Path("O2021_84_41_1_0003_00074400_7000_3000.tif"),Path("O2021_84_40_1_0051_00071950_7000_4000.tif"),Path("O2021_84_40_1_0051_00071950_4000_1000.tif"),Path("O2021_84_40_1_0051_00071950_3000_2000.tif"),Path("O2021_84_40_1_0051_00071950_2000_0.tif")]
    industri =  []



    image_paths_for_subsets= {}
    label_paths_for_subsets = {}
    prediction_paths_for_subsets = {}

    #images
    #image_paths_for_subsets["images_with_ignore"]= [images_base_path/im for im in images_with_ignore]
    image_paths_for_subsets["SUBURB"]= [images_base_path/im for im in forstad]
    image_paths_for_subsets["URBAN"]= [images_base_path/im for im in bykerne]
    image_paths_for_subsets["RURAL"] = [images_base_path / im for im in land]
    image_paths_for_subsets["INDUSTRY"] = [images_base_path / im for im in industri]
    image_paths_for_subsets["PATHS"] = [images_base_path / im for im in paths]

    #labels
    #label_paths_for_subsets["images_with_ignore"]= [labels_base_path/im for im in images_with_ignore]
    label_paths_for_subsets["SUBURB"]= [labels_base_path/im for im in forstad]
    label_paths_for_subsets["URBAN"]= [labels_base_path/im for im in bykerne]
    label_paths_for_subsets["RURAL"] = [labels_base_path / im for im in land]
    label_paths_for_subsets["INDUSTRY"] = [labels_base_path / im for im in industri]
    label_paths_for_subsets["PATHS"] = [labels_base_path / im for im in paths]


    #SUBURB, URBAN, RURAL, INDUSTRY, LAKES AND STREAMS, COASTS, PORTS, ROADS, WELL KNOWN PLACES
    #predictions
    #prediction_paths_for_subsets["images_with_ignore"]= [prediction_base_path/im for im in images_with_ignore]
    prediction_paths_for_subsets["SUBURB"]= [prediction_base_path/im for im in forstad]
    prediction_paths_for_subsets["URBAN"]= [prediction_base_path/im for im in bykerne]
    prediction_paths_for_subsets["RURAL"] = [prediction_base_path / im for im in land]
    prediction_paths_for_subsets["INDUSTRY"] = [prediction_base_path / im for im in industri]
    prediction_paths_for_subsets["PATHS"] = [prediction_base_path / im for im in paths]



    result={"images":image_paths_for_subsets,"labels":label_paths_for_subsets,"predictions":prediction_paths_for_subsets}
    return result
