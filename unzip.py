import zipfile
with zipfile.ZipFile('/datasets/Paderborn/segmented/vibration/vibration.zip', 'r') as zip_ref:
    zip_ref.extractall('/datasets/Paderborn/segmented/vibration/')