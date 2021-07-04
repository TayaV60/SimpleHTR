import argparse
from path import Path
import json
import os
import glob

class ConverstionFilePaths:
    """Filenames and paths to data."""
    annotation_files = 'ann'
    image_files = 'img'
    image_extension = 'jpg'

def iterate_json_files(directory):
  image_data_list = []
  for filepath in sorted(glob.glob(directory + '/*.json')):
    if filepath.endswith(".json"): 
        filepath = os.path.join(filepath)
        with open(filepath) as jsonFile:
          data = json.load(jsonFile)
          filename = os.path.basename(filepath)
          file_without_extension = os.path.splitext(filename)[0]
          image_data = {
            "id": file_without_extension,
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
        line = item["id"] + " ok 154 0 0 " + str(item["width"]) + " " + str(item["height"]) + " " + item["description"]
        f.write("%s\n" % line)


def main():
    """Main conversion function (converts Russian dataset to IAM format)."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_folder', help='Folder to convert.', type=Path, default='/Users/taisiyavelarde/Desktop/DatasetRus/Words/20200923_Dataset_Words_Public')
    parser.add_argument('--destination_folder', help='Folder to store converted files.', type=Path, default='/Users/taisiyavelarde/Documents/tmp')

    args = parser.parse_args()

    print(args.destination_folder)

    assert args.original_folder.exists()
    assert args.destination_folder.exists()

    print(args.original_folder + '/' + ConverstionFilePaths.annotation_files)

    annotation_files_folder = Path(args.original_folder + '/' + ConverstionFilePaths.annotation_files)
    image_files_folder =  Path(args.original_folder + '/' + ConverstionFilePaths.image_files)

    assert annotation_files_folder.exists()
    assert image_files_folder.exists()

    image_data_list = iterate_json_files(args.original_folder + '/' + ConverstionFilePaths.annotation_files)
    save_words_txt(image_data_list, args.destination_folder)


if __name__ == '__main__':
    main()
