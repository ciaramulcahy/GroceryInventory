import argparse
import re
#import nltk

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description= 'Process OCR text files of receipts')
ap.add_argument('filename', help="path to input text file to be parsed")
ap.add_argument('store', help="1-Kroger, 2-Trader Joe's)
args = ap.parse_args()

# Could add find store number later, if feel like shopping multiple places

def find_store(store):
        #This is not elegant whatsoever, dropdown like in excel would be more effective
        Kroger = False
        TJs = False
        if store == 1:
                Kroger = True
        if store == 2:
                TJs = True
                


def list_items(cleanlines):
	now_items = False
	items = []
	lookup = 'DA'
	lookupAlt = 'LY'
	enditems = 'SUB'
	enditemsAlt = 'TOT'

	for line in cleanlines:
		if enditems in line or enditemsAlt in line:
			now_items = False
		if now_items:
			items.append(line)
		if lookup in line or lookupAlt in line:
			now_items = True
	return items

def item_price_tuple(line):
	# to handle multiline item names
                # Have considered iterating in reverse and saving prior item name temporarily
	#return a tuple of (item_name, price)
	item = ""
	price = ""
	priceOngoing = True
	for char in reversed(line):
		if char.isalpha():
			priceOngoing = False
			item += char
		elif priceOngoing:
			price += char
		else:
			item +=char
	item = ''.join(reversed(item)).strip()
	price = ''.join(reversed(price)).strip()

	# Later Improvement: Measure distance between item name an item names in some list/database 
	# Evaluate if ought to just replace read item name with existing item name
                
	# Not sure whether to segment by word (split then join) or as an entire text
	
	# Make sure do not return non-item lines, "You saved" lines might still be included though, filter later
	if price =='' or item =='':
		return				# Return None
	else:
		return (item, price)
	# Check than item and price are reasonable formats, like not weird characters
                
def item_price_tuple_nltk(line):	# Work in progress; line seems to be iterable like a list
	# Let's use the nltk reg expression default
	pattern = r'''(?x)(?:[A-Z]\.) + | \w+(?:-\w+) * | \$?\d+(?:\.\d+)?%? | \.\.\.| [][.,;"'?():-_`]'''   
	regExp = nltk.regexp_tokenize(line, pattern)
	if len(regExp) ==2:
		item = regExp[0]
		price = regExp[1]
		# Clean up item to see if it matches the text common on receipts + relate to FDC descriptions
		# clean up price to have one decimal and only numbers, or throw an exception + prompt for fixing
		return (item, price)

def make_items_list_Kroger(lines):
	#excluded_words = {}
	'''with open(args.filename, 'r') as f:
					lines = f.readlines()'''
	#date handling	
	dateNext = False
	date = ""

	#list making
	cleanlines = []
	TJs = False

	# Find date and remove whitespace lines
	for line in lines:
		# find what date transaction happened, if scan good enough in that area of receipt
		if dateNext:
			print ("dateNext is True")
			for char in line[0:8:1]:
				date += char
			print (date)
			dateNext = False
		elif "ITEMS" in line:
			print ("Found ITEMS")
			dateNext = True
			continue

		# Make list cleanlines of the non-blank lines
		if not line.strip():
			continue
		else:
			cleanlines.append(line)
		
	#print (cleanlines)
	for line in cleanlines:
		# Identify the grocery store
		if "TRADER" in line:
			print ("TRADER JOE'S confirmed")
			TJs = True
		#print(line)		#Display the text file
			
if Kroger == True:
	# form a set of (item, price) tuples included on the list
	item_price_list = []
	items_only = list_items(cleanlines)
	#print (TJs_list_items(cleanlines))
	for line in items_only:
		entry = item_price_tuple(line)
		if entry != None:
			item_price_list.append(entry)
	#print(item_price_list)
	return item_price_list, date


# Insert the quantity of each food item, associated with the date, into a database


# To run this: python3 read_Kroger.py RT3_output.TXT
