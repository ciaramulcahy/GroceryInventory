# GroceryInventory

# Perform OCR
A photo is taken of a food receipt. 
Upon uploading the photo, the photo is transformed into a rectangular section  and the area around the receipt is cropped to reduce noise and improve contrast of the text on the receipt (transform.py).
Optical character recognition(OCR) is performed on the photo to transcribe the text from the receipt into a text file.
Alternatively, the text can be copied from a photo by LiveText (iPhone ios 15 or later) or Google Lens (Android) and pasted into a text file.

# Parse Receipt Text
As receipt content is arranged slightly differently between grocery store companies, 
the text file of the receipt interpretted using a program specific to the grocery store chain.
The file naming format is parse_StoreName.py

Information extracted from the grocery receipt will include the following:
- UPC (Universal Product Code, if included on receipt)
- Item name
- Price
- Date, Location (if included on receipt)

The text file of the receipt likely contains errors. 
If the string is similar enough to a known receipt item entry, the entry will be corrected and item entered into the customer's database of food items.

# Relate to External Data Sources
This food item name listed on a receipt from a given grocery store can be related to the FDA Food Data Central database to include the following information:
- Commodity food
- Quantity
- Nutrition info

That global information can be used to calculate and compare the following factors across brands and stores: 
- unit price ($/lb, $/ounce, $ / fl. ounce)
- Price per different nutrients (Calorie, gram of protein etc)

# Reports
- Total spend per category (snacks, fruit, vegetables, carbs, alcohol, beverages)
- Nutritional deficiencies, presence of allergens/sensitivities
