import csv

def write_csv_l(data):
    with open('example.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

