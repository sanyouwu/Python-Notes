from scipy.io import arff
from unrar import rarfile

'''
Extract .rar files through Python
'''
img_dir = r"D:\Codes\IndividualizedRegionSelectionMIL\raw_data\MILDATA"
for file in os.listdir(img_dir):
    if file.split(".")[-1] == "rar":
        name = rarfile.RarFile(os.path.join(img_dir,file))
        name.extractall(img_dir)