import zipfile
with zipfile.ZipFile('datasets.zip', 'r') as zip_ref:
    zip_ref.extractall()