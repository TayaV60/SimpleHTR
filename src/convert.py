import argparse
import random
from path import Path
import pathlib
import datetime
import json
import os
import glob
import cv2
import numpy
from matplotlib import pyplot as plt
import sys

class ConverstionFilePaths:
    """Filenames and paths to data."""
    image_files = 'img' # same for RUS and IAM
    input_annotation_files = 'ann'
    output_annotation_files = 'gt'
    image_extension = 'jpg'
    output_extenstion = 'png'

def get_randomised_json_file_list(directory):
  filepaths = []
  for filepath in sorted(glob.glob(directory + '/*.json')):
    if filepath.endswith(".json"): 
        filepath = os.path.join(filepath)
        filepaths.append(filepath)
  random.shuffle(filepaths)
  return filepaths

def iterate_json_files(directory):
  filepaths = get_randomised_json_file_list(directory)
  image_data_list = []
  bucket_size = 50
  buckets = numpy.array_split(filepaths, bucket_size)
  for x, bucket_files in enumerate(buckets):
    sub_buckets = numpy.array_split(bucket_files, bucket_size)
    for y, sub_bucket_files in enumerate(sub_buckets):
      for z, filepath in enumerate(sub_bucket_files):
        with open(filepath) as jsonFile:
              data = json.load(jsonFile)
              filename = os.path.basename(filepath)
              file_without_extension = os.path.splitext(filename)[0]
              file_dashed = str(x) + "-" + str(y) + '-' + str(z)
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
        line = item["destination_id"] + " ok 154 0 0 " + str(item["width"]) + " " + str(item["height"]) + " AT " + item["description"]
        f.write("%s\n" % line)

def save_json_file_list(image_data_list, destination_folder) -> None:
  """Save the json file list for mapping reference to the original files."""
  with open(destination_folder + '/' + 'word_list.json', 'w') as f:
      json.dump({'date': str(datetime.datetime.now()), 'image_data': image_data_list}, f)

def read_json_file_list(destination_folder) -> None:
  """Save the json file list for mapping reference to the original files."""
  with open(destination_folder + '/' + 'word_list.json', 'r') as f:
    data = json.load(f)
    return data["image_data"]

def gauss_image(img, sigma):
  return cv2.GaussianBlur(img,(sigma,sigma),0)

def thresh_image(img):
  # see https://docs.opencv.org/4.5.2/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
  _, mask = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
  return mask

# see https://www.programmersought.com/article/32324773302/
def gamma_image(img, gamma):
  gammad = numpy.power(img/float(numpy.max(img)), 1/gamma)
  return cv2.convertScaleAbs(gammad, alpha=(255.0))

# using https://www.pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
def save_hist(img, title, destination_folder, file_name):
  plt.subplot(1,2,1)
  plt.imshow(img,'gray')
  plt.title(f"{title} Image")
  plt.xticks([])
  plt.yticks([])

  plt.subplot(1,2,2)
  plt.hist(img.ravel(),50,[0,256])
  plt.title(f"{title} Hist")
  plt.xticks(numpy.arange(0, 256, step=50))

  plt.savefig(f"{destination_folder}/${file_name}")
  plt.close()

def convert_image(input_file, destination_folder, output_file, debug):
  img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
  
  gam = gamma_image(img, 0.5)
  mask = thresh_image(gam)
  # masking adapted from https://www.analyticsvidhya.com/blog/2019/03/opencv-functions-computer-vision-python/
  # and https://medium.com/featurepreneur/performing-bitwise-operations-on-images-using-opencv-6fd5c3cd72a7
  masked = cv2.bitwise_not(gam, mask=mask)
  # adapted from https://stackoverflow.com/a/40954142
  uninverted = cv2.bitwise_not(masked)
  blurred = gauss_image(uninverted, 3)

  ext = ConverstionFilePaths.output_extenstion

  if (debug):
    save_hist(img, f"{output_file} Input", destination_folder, f"{output_file}-1-input.hist.png")
    save_hist(gam, f"{output_file} Gamma", destination_folder, f"{output_file}-2-gam.hist.png")
    save_hist(mask, f"{output_file} Mask", destination_folder, f"{output_file}-3-mask.hist.png")
    save_hist(masked, f"{output_file} Masked", destination_folder, f"{output_file}-4-masked.hist.png")
    save_hist(uninverted, f"{output_file} Uninverted", destination_folder, f"{output_file}-5-uninverted.hist.png")
    save_hist(blurred, f"{output_file} Blurred", destination_folder, f"{output_file}-6-blurred.hist.png")
    cv2.imwrite(f"{destination_folder}/{output_file}-1-gam.{ext}", img)
    cv2.imwrite(f"{destination_folder}/{output_file}-2-gam.{ext}", gam)
    cv2.imwrite(f"{destination_folder}/{output_file}-3-mask.{ext}", mask)
    cv2.imwrite(f"{destination_folder}/{output_file}-4-masked.{ext}", masked)
    cv2.imwrite(f"{destination_folder}/{output_file}-5-uninverted.{ext}", uninverted)
    cv2.imwrite(f"{destination_folder}/{output_file}-6-blurred.{ext}", blurred)

  cv2.imwrite(f"{destination_folder}/{output_file}.{ext}", blurred)

def convert_images(image_data_list, original_image_folder, base_destination_folder, limit, debug):
  for i, item in enumerate(image_data_list):
    input_file = original_image_folder + "/" + item["id"] + "." + ConverstionFilePaths.image_extension
    destination_id_split = item["destination_id"].split("-")
    file_name_subdir = destination_id_split[0]
    file_name_subdir2 = f'{destination_id_split[0]}-{destination_id_split[1]}'
    destination_folder = base_destination_folder + '/' + file_name_subdir + '/' + file_name_subdir2
    # create the folder if it does not exist
    pathlib.Path(destination_folder).mkdir(parents=True, exist_ok=True)
    output_file = item["destination_id"]
    # print ("Writing " + input_file + " -> " + output_file)
    convert_image(input_file, destination_folder, output_file, debug)
    if (i > limit):
      return
    

def main():
    """Main conversion function (converts Russian dataset to IAM format)."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_folder', help='Folder to convert.', type=Path, default='/Users/taisiyavelarde/Desktop/DatasetRus/Words/20200923_Dataset_Words_Public')
    parser.add_argument('--destination_folder', help='Folder to store converted files.', type=Path, default='/Users/taisiyavelarde/Documents/tmp')
    parser.add_argument('--words_only', help='Do not regenerate images.', action='store_true')
    parser.add_argument('--regenerate_file_list', help='Regenerate words and json file list.', action='store_false')
    parser.add_argument('--limit', help='Limit the number of images processed.', type=int, default=sys.maxsize)
    parser.add_argument('--debug', help='Generate debug histograms.', action='store_true')

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

    if (args.regenerate_file_list):
      print("Regenerating data list")
      image_data_list = iterate_json_files(input_annotation_files_folder)
      save_words_txt(image_data_list, output_annotation_files_folder)
      save_json_file_list(image_data_list, args.destination_folder)
    else:
      print("Reading data list")
      image_data_list = read_json_file_list(args.destination_folder)

    if args.words_only != True:
      print(f"Converting images: Limit? {args.limit} Debug? {args.debug}")
      convert_images(image_data_list, input_image_files_folder, output_image_files_folder, args.limit, args.debug)


if __name__ == '__main__':
    print("STARTED")
    print(datetime.datetime.now())
    main()
    print("ENDED")
    print(datetime.datetime.now())
