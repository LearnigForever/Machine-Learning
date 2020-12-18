# region python_package_imports
import os
import cv2
import sys
import shutil
import numpy as np

sys.path.append('./../')
from utils.common_utils import get_logger
# endregion

class DataPreparerSegmentation:

    def __init__(self):
        self.base_input_path = "../../Data/training/Segmentation/input/"
        self.base_output_path = "../../Data/training/Segmentation/output/"
        self.raw_masks_folder = os.path.join(self.base_output_path, "Raw_Masks")
        self.encoded_masks_folder = os.path.join(self.base_output_path, "Encoded_Masks")

    def _create_folders(self):
        if not os.path.exists(self.raw_masks_folder):
            os.makedirs(self.raw_masks_folder)

        if not os.path.exists(self.encoded_masks_folder):
            os.makedirs(self.encoded_masks_folder)

    def list_all_folders(self):
        lst_folders = os.listdir(self.base_input_path)
        return lst_folders

    def encode_images_for_label_map(self, label_class_code):
        """
           Description: This function uses label map dict and original segmentation masks and
           saves encoded masks

        """
        log.info(label_class_code)

        list_raw_mask_files = os.listdir(self.raw_masks_folder)
        file_counter = 1
        for raw_mask_file in list_raw_mask_files:
             log.info("Processing {} - file {}".format(file_counter, raw_mask_file))
             img = cv2.imread(os.path.join(self.raw_masks_folder, raw_mask_file))

             if not isinstance(img, np.ndarray):
                 log.error("{} is read as None by cv2 read".format(raw_mask_file))
                 continue

             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             img_encoded = img.copy()

             for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_encoded[i][j] = label_class_code[tuple(img[i][j])][1]
             cv2.imwrite(os.path.join(self.encoded_masks_folder, raw_mask_file), img_encoded)
             file_counter = file_counter + 1

    def _copy_masks_in_raw_masks_folder(self, folders_list):
        for f in folders_list:
            seg_class_path = os.path.join(self.base_input_path, f)
            seg_class_path = os.path.join(seg_class_path, "SegmentationClass")
            all_files = os.listdir(seg_class_path)

            for file in all_files:
                source_file = os.path.join(seg_class_path, file)
                splits = file.split(".")
                splits_len = len(splits)
                base_file_name = splits[0:splits_len - 1]
                base_file_name = ".".join(base_file_name)
                shutil.copy(source_file, os.path.join(self.raw_masks_folder, base_file_name + ".jpg"))

    def get_label_map(self, folders_list, codes, code_encoding):
        """
            Description:
            This function reads all label maps for segmentation and creates dictionary for
            segmentation encoding listing only relevamt codes
        """

        all_lines = []
        for f in folders_list:
            label_map_path = os.path.join(self.base_input_path, f)
            label_map_path = os.path.join(label_map_path, "labelmap.txt")

            if not os.path.exists(label_map_path):
                log.error("label map path does not exist - {}".format(label_map_path))
                continue

            label_map_file = open(label_map_path, 'r')

            for line in label_map_file:

                # if line is a comment, continue
                if line.startswith('#'):
                    continue

                line_substr = line.strip().split("::")[0]
                current_code = line_substr.split(":")[0]
                if current_code.lower() in codes:
                    all_lines.append(line_substr)
            label_map_file.close()

        # get unique code lines
        all_lines = list(set(all_lines))
        if len(all_lines) != len(codes):
            log.error("Please check the label map codes.")

        label_class_code = {}
        encoding_counter = 0
        for line in all_lines:
            class_name = line.split(":")[0]
            code_index = codes.index(class_name)

            color_tuple = line.split(":")[1]
            color_tuple = color_tuple.split(',')
            color_tuple = tuple([int(i) for i in color_tuple])
            encoding = (code_encoding[code_index], code_encoding[code_index], code_encoding[code_index])
            encoding_counter = encoding_counter + 1
            label_class_code[color_tuple] = [class_name, encoding]

        log.info(label_class_code)

        return label_class_code


if __name__ == "__main__":

    # iterate through output folder

    log = get_logger(__name__)

    data_preparer_obj = DataPreparerSegmentation()

    # Create "Raw_Masks" and "Encoded_Masks" folders
    data_preparer_obj._create_folders()

    folders_list = data_preparer_obj.list_all_folders()

    codes = ['background', 'jsv', 'rsv']
    encodings = [0, 3, 4]
    label_class_code = data_preparer_obj.get_label_map(folders_list=folders_list,
                                                       codes=codes,
                                                       code_encoding=encodings)

    # Copy all Masks in Raw_Masks folder
    data_preparer_obj._copy_masks_in_raw_masks_folder(folders_list)

    # Encode the image and save the encoded masks in Encoded_Masks folder
    data_preparer_obj.encode_images_for_label_map(label_class_code=label_class_code)
