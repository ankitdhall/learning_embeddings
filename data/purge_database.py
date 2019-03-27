import argparse
import json
import os
from tqdm import tqdm
import cv2


def purge_image_data(data, images_dir, clean_dir):
    for data_id in tqdm(data):
        sample = data[data_id]
        if sample['family'] not in ['Papilionidae', 'Pieridae', 'Nymphalidae', 'Lycaenidae', 'Hesperiidae', 'Riodinidae']:
            continue
        if sample['image_path'] == "" or '.JPG' in sample['image_path']:
            correct_folder_name = sample['image_name'][11:21] + "R"
            path_to_image = os.path.join(images_dir, correct_folder_name, sample['image_name'])
            path_to_clean_dir = os.path.join(clean_dir, correct_folder_name)
        else:
            path_to_image = os.path.join(images_dir, sample['image_path'], sample['image_name'])
            path_to_clean_dir = os.path.join(clean_dir, sample['image_path'])
        if not os.path.exists(path_to_clean_dir):
            os.makedirs(path_to_clean_dir)

        if os.path.exists(os.path.join(path_to_clean_dir, sample['image_name'])):
            continue
        img = cv2.imread(path_to_image)
        cv2.imwrite(os.path.join(path_to_clean_dir, sample['image_name']), img)


def purge_json_data(data, json_path):
    purged_data = {}

    for data_id in tqdm(data):
        sample = data[data_id]
        if sample['family'] not in ['Papilionidae', 'Pieridae', 'Nymphalidae', 'Lycaenidae', 'Hesperiidae', 'Riodinidae']:
            continue
        purged_data[data_id] = sample

    with open(json_path, 'w') as outfile:
        json.dump(purged_data, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", help='Use the mini database for testing/debugging.', action='store_true')
    parser.add_argument("--images_dir", help='Parent directory with images.', type=str, required=True)
    parser.add_argument("--clean_dir", help='Clean directory with images.', type=str, required=True)
    parser.add_argument("--json_path", help='Path to json with relevant data.', type=str, required=True)
    args = parser.parse_args()

    infile = 'database'
    if args.mini:
        infile = 'mini_database'

    if os.path.isfile('../database/{}.json'.format(infile)):
        with open('../database/{}.json'.format(infile)) as json_file:
            data = json.load(json_file)
    else:
        print("File does not exist!")
        exit()

    # purge_image_data(data, args.images_dir, args.clean_dir)
    # purge_json_data(data, args.json_path)
