import argparse
from path import Path
import pathlib
import datetime
import json
import os
import glob
import cv2

class ConverstionFilePaths:
    """Filenames and paths to data."""
    image_files = 'img' # same for RUS and IAM
    input_annotation_files = 'ann'
    output_annotation_files = 'gt'
    image_extension = 'jpg'
    output_extenstion = 'png'

def iterate_json_files(directory):
  image_data_list = []
  for filepath in sorted(glob.glob(directory + '/*.json')):
    if filepath.endswith(".json"): 
        filepath = os.path.join(filepath)
        with open(filepath) as jsonFile:
          data = json.load(jsonFile)
          filename = os.path.basename(filepath)
          file_without_extension = os.path.splitext(filename)[0]
          file_dashed = file_without_extension.replace("_", "-")
          image_data = {
            "id": file_without_extension,
            "destination_id": file_dashed,
            "width": data["size"]["width"],
            "height": data["size"]["height"],
            "description": data["description"]
          }
          image_data_list.append(image_data)
          # print(file_without_extension + " ok 154 0 0 " + str(data["size"]["width"]) + " " + str(data["size"]["height"])  + " " + data["description"])
  return image_data_list

def save_words_txt(image_data_list, destination_folder):
  with open(destination_folder + '/words.txt', 'w') as f:
    for item in image_data_list:
        line = item["destination_id"] + " ok 154 0 0 " + str(item["width"]) + " " + str(item["height"]) + " " + item["description"]
        f.write("%s\n" % line)

def convert_image(input_file, output_file):
  img = cv2.imread(input_file)
  cv2.imwrite(output_file, img)

def convert_images(image_data_list, original_image_folder, base_destination_folder):
  for item in image_data_list:
    input_file = original_image_folder + "/" + item["id"] + "." + ConverstionFilePaths.image_extension
    destination_id_split = item["destination_id"].split("-")
    destination_folder = base_destination_folder + '/' + destination_id_split[0] + '/' + destination_id_split[1]
    # create the folder if it does not exist
    pathlib.Path(destination_folder).mkdir(parents=True, exist_ok=True)
    output_file = destination_folder + "/" + item["destination_id"] + "." + ConverstionFilePaths.output_extenstion
    # print ("Writing " + input_file + " -> " + output_file)
    convert_image(input_file, output_file)
    

def main():
    """Main conversion function (converts Russian dataset to IAM format)."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_folder', help='Folder to convert.', type=Path, default='/Users/taisiyavelarde/Desktop/DatasetRus/Words/20200923_Dataset_Words_Public')
    parser.add_argument('--destination_folder', help='Folder to store converted files.', type=Path, default='/Users/taisiyavelarde/Documents/tmp')

    args = parser.parse_args()

    # verify input folders
    print("Reading from " + args.original_folder)
    assert args.original_folder.exists()
    input_annotation_files_folder = args.original_folder + '/' + ConverstionFilePaths.input_annotation_files
    input_image_files_folder = args.original_folder + '/' + ConverstionFilePaths.image_files
    assert Path(input_annotation_files_folder).exists()
    assert Path(input_image_files_folder).exists()

    print("Writing to " + args.destination_folder)
    # create output folders
    output_annotation_files_folder = args.destination_folder + '/' + ConverstionFilePaths.output_annotation_files
    output_image_files_folder = args.destination_folder + '/' + ConverstionFilePaths.image_files
    pathlib.Path(output_annotation_files_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_image_files_folder).mkdir(parents=True, exist_ok=True)

    image_data_list = iterate_json_files(input_annotation_files_folder)
    save_words_txt(image_data_list, output_annotation_files_folder)

    convert_images(image_data_list, input_image_files_folder, output_image_files_folder)


if __name__ == '__main__':
    print("STARTED")
    print(datetime.datetime.now())
    main()
    print("ENDED")
    print(datetime.datetime.now())
