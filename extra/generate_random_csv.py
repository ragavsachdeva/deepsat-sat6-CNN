from random import randint
import argparse

filename = "test.csv"
rows = 10
cols = 10

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--Filename", help="Please provide the name of the csv file.")
parser.add_argument("-r", "--Rows", help="Please provide the number of rows in the csv file.")
parser.add_argument("-c", "--Cols", help="Please provide the number of columns.")
args = parser.parse_args()

if args.Filename:
    filename = args.Filename
if args.Rows:
    rows = int(args.Rows)
if args.Cols:
    cols = int(args.Cols)

with open(filename, "w") as myfile:
	myfile.write("")

with open(filename, "a") as myfile:
	
	for x in range(rows):
		row_text = ""
		row_text += str(randint(0,255))
		for y in range(cols-1):
			row_text+=","
			row_text += str(randint(0,255))
		row_text+='\n'
		myfile.write(row_text)