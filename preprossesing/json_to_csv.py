import json
import csv

with open('data.json', 'r', encoding = 'utf-8') as input_file, open('data.csv', 'w', newline = '') as output_file :
    data = json.load(input_file)
    f = csv.writer(output_file)
    f.writerow(["cls_id", "L_Category", "M_Category", "S_Category", "year", "color", "direction", "file_name"])
    for datum in data:
        f.writerow([datum["cls_id"], datum["L_Category"], datum["M_Category"], datum["S_Category"], datum["year"], datum["color"], datum["direction"], datum["file_name"]])
