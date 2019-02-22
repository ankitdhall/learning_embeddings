from openpyxl import load_workbook
import json
import uuid

wb = load_workbook("../database/Metadata.xlsx")
sheet = wb.active

n_samples = 2

field_access = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
field_names = ['image_path', 'family', 'subfamily', 'tribe', 'genus', 'specific_epithet', 'subspecific_epithet',
               'infraspecific_epithet', 'infraspecific_rank', 'author']

data = {}
for sample_id in range(2, n_samples + 2):
    db_element = {}
    token = str(uuid.uuid4())
    db_element['token'] = token
    for field_access_key, field_name in zip(field_access, field_names):
        if field_name == 'image_path':
            db_element[field_name] = sheet['{}{}'.format(field_access_key, sample_id)].value[1:]
        else:
            db_element[field_name] = sheet['{}{}'.format(field_access_key, sample_id)].value
    data[token] = db_element

with open('database.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)
