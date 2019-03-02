from openpyxl import load_workbook
import json
import uuid
from tqdm import tqdm

wb = load_workbook("../database/allspecimens.xlsx")
sheet = wb.active

n_samples = 197052

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

with open('../database/database.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
