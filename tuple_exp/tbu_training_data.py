import csv
import ast

with open("tuple_exp/TBU","r") as fin, open("tuple_exp/tuple_dataset.csv", "w") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    _ = next(reader)
    for row in reader:
        line = ast.literal_eval(row[4])
        writer.writerow(line)
