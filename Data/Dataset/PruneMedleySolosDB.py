from tqdm import tqdm

from TorchJaekwon.Util import UtilData, Util

meta_data_list = UtilData.walk('Data/Dataset/MedleySolosDB')
rm_file_path_list = [meta['file_path'] for meta in meta_data_list if meta['file_name'].startswith('.')]
for rm_file_path in tqdm(rm_file_path_list):
    Util.system(f'rm -rf {rm_file_path}')
