import argparse
from path import Path
import json
import os

class ConverstionFilePaths:
    """Filenames and paths to data."""
    annotation_files = 'ann'
    image_files = 'img'
    image_extension = 'jpg'

def iterate_json_files(directory):
  for filename in os.listdir(directory):
    if filename.endswith(".json"): 
        filepath = os.path.join(directory, filename)
        with open(filepath) as jsonFile:
          data = json.load(jsonFile)
          file_without_extension = os.path.splitext(filename)[0]
          print(file_without_extension + " " + str(data["size"]["width"]) + " " + str(data["size"]["height"])  + " " + data["description"])
        continue
    else:
        continue

def main():
    """Main conversion function (converts Russian dataset to IAM format)."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_folder', help='Folder to convert.', type=Path, default='/Users/taisiyavelarde/Desktop/DatasetRus/Words/20200923_Dataset_Words_Public')
    parser.add_argument('--destination_folder', help='Folder to store converted files.', type=Path, default='/Users/taisiyavelarde/Documents/tmp')

    args = parser.parse_args()

    print(args.destination_folder)

    assert args.original_folder.exists()
    assert args.destination_folder.exists()

    annotation_files_folder = Path(args.original_folder + '/' + ConverstionFilePaths.annotation_files)
    image_files_folder =  Path(args.original_folder + '/' + ConverstionFilePaths.image_files)

    assert annotation_files_folder.exists()
    assert image_files_folder.exists()

    iterate_json_files(annotation_files_folder)

if __name__ == '__main__':
    main()
