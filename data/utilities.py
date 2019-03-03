from openpyxl import load_workbook
import json
import uuid
from tqdm import tqdm
import argparse

def generate_database(create_mini=False):
    """
    Read .xlsx file and generate a .json database.
    :param create_mini: To create a smaller version of the database for debugging.
    :return: None
    """
    wb = load_workbook("../database/allspecimens.xlsx")
    sheet = wb.active

    n_samples = 197052
    if create_mini:
        n_samples = 10
        print('Generating mini database with {}'.format(n_samples))

    field_access = ['BK', 'BJ', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'AH', 'AI', 'AP', 'AQ', 'A']
    field_names = ['image_path', 'image_name', 'family', 'subfamily', 'genus', 'specific_epithet', 'subspecific_epithet',
                   'infraspecific_epithet', 'author', 'country', 'primary_division', 'dec_lat', 'dec_long', 'barcode']

    data = {}
    for sample_id in tqdm(range(2, n_samples + 2)):
        db_element = {}
        token = str(uuid.uuid4())
        db_element['token'] = token
        for field_access_key, field_name in zip(field_access, field_names):
            if field_name == 'image_path':
                db_element[field_name] = sheet['{}{}'.format(field_access_key, sample_id)].value[1:].strip().strip('\\')
            elif field_name in ['dec_lat', 'dec_long']:
                try:
                    db_element[field_name] = float(sheet['{}{}'.format(field_access_key, sample_id)].value.strip())
                except ValueError:
                    db_element[field_name] = None
            else:
                db_element[field_name] = sheet['{}{}'.format(field_access_key, sample_id)].value.strip()
        data[token] = db_element

    db_file = 'database'
    if create_mini:
        db_file = 'mini_database'
    with open('../database/{}.json'.format(db_file), 'w') as outfile:
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", help='Generate a mini database for testing/debugging.', action='store_true')
    args = parser.parse_args()

    generate_database(args.mini)

