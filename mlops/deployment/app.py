import streamlit as st
import pandas as pd
import re
from datetime import date
from huggingface_hub import hf_hub_download
import joblib

# load the model
repo_id = "JaiBhatia020373/mlops"
repo_type = "model"

model_path = hf_hub_download(repo_id="JaiBhatia020373/mlops", 
                             filename="sales-forecast.joblib", 
                             repo_type="model")

model = joblib.load(model_path)

preprocessor_path = hf_hub_download(repo_id="JaiBhatia020373/mlops", 
                             filename="preprocessor.joblib", 
                             repo_type="model")

preprocessor = joblib.load(preprocessor_path)

st.title("SuperKart Sales Forecast Prediction")

# define a dictionary of product weights, organized by product_types and products
# all weights are in kg
product_weights = {
    "Meat": {"Chicken": "12.0", "Mutton": "15.0"},
    "Snack foods": {"Potato chips": "15.0", "Chocolate bar": "0.05"},
    "Hard drinks": {"Whiskey": "0.7", "Vodka": "0.7"},
    "Dairy": {"Milk": "15.0", "Cheese": "9.0"},
    "Canned": {"Canned beans": "8.0", "Canned tuna": "15.0"},
    "Soft drinks": {"Cola": "18.0", "Lemon soda": "7.0"},
    "Health and hygiene": {"Soap": "9.5", "Toothpaste": "16.0"},
    "Baking goods": {"Flour": "20.0", "Baking powder": "2.0"},
    "Bread": {"White bread": "0.4", "Whole wheat bread": "0.5"},
    "Breakfast": {"Cornflakes": "9.0", "Oats": "16.0"},
    "Frozen foods": {"Frozen peas": "0.5", "Ice cream": "0.5"},
    "Fruits and vegetables": {"Apple": "15.0", "Tomato": "9.0"},
    "Household": {"Detergent": "15.0", "Dishwashing liquid": "8.0"},
    "Seafood": {"Prawns": "0.25", "Salmon": "0.2"},
    "Starchy foods": {"Rice": "15.0", "Pasta": "8.0"},
    "Others": {"Batteries": "0.1", "Stationary": "0.2"}
}

product_sugar_content = {
    "Meat": {"Chicken": "Low sugar", "Mutton": "Regular"},
    "Snack foods": {"Potato chips": "Low sugar", "Chocolate bar": "Regular"},
    "Hard drinks": {"Whiskey": "Low sugar", "Vodka": "Low sugar"},
    "Dairy": {"Milk": "Low sugar", "Cheese": "Regular"},
    "Canned": {"Canned beans": "Low sugar", "Canned tuna": "Low sugar"},
    "Soft drinks": {"Cola": "Low sugar", "Lemon soda": "Low sugar"},
    "Health and hygiene": {"Soap": "No Sugar", "Toothpaste": "No Sugar"},
    "Baking goods": {"Flour": "Low sugar", "Baking powder": "Low sugar"},
    "Bread": {"White bread": "Regular", "Whole wheat bread": "Low sugar"},
    "Breakfast": {"Cornflakes": "Regular", "Oats": "Low sugar"},
    "Frozen foods": {"Frozen peas": "Low sugar", "Ice cream": "Regular"},
    "Fruits and vegetables": {"Apple": "Regular", "Tomato": "Low sugar"},
    "Household": {"Detergent": "No Sugar", "Dishwashing liquid": "No Sugar"},
    "Seafood": {"Prawns": "Low sugar", "Salmon": "Low sugar"},
    "Starchy foods": {"Rice": "Low sugar", "Pasta": "Low sugar"},
    "Others": {"Batteries": "No Sugar", "Stationary": "No Sugar"}   
}

product_codes = {
    "Chicken": "FD3834",
    "Mutton": "FD175",
    "Potato chips": "FD5680",
    "Chocolate bar": "FD2652",
    "Whiskey": "DR4728",
    "Vodka": "DR1983",
    "Milk": "FD7839",
    "Cheese": "FD3355",
    "Canned beans": "FD7662",
    "Canned tuna": "FD32",
    "Cola": "DR4015",
    "Lemon soda": "DR2519",
    "Soap": "NC1897",
    "Toothpaste": "NC177",
    "Flour": "FD4450",
    "Baking powder": "FD7559",
    "White bread": "FD6692",
    "Whole wheat bread": "FD5710",
    "Cornflakes": "FD4479",
    "Oats": "FD7154",
    "Frozen peas": "FD6114",
    "Ice cream": "FD5743",
    "Apple": "FD1067",
    "Tomato": "FD3750",
    "Detergent": "NC2141",
    "Dishwashing liquid": "NC278",
    "Prawns": "FD6616",
    "Salmon": "FD5645",
    "Rice": "FD5982",
    "Pasta": "FD2413",
    "Batteries": "NC3408",
    "Stationary": "NC4605"
}

product_MRP_prices = {
    "Chicken": "120.5",
    "Mutton": "210.3",
    "Potato chips": "10.0",
    "Chocolate bar": "20.0",
    "Whiskey": "670.0",
    "Vodka": "400.0",
    "Milk": "60.0",
    "Cheese": "150.0",
    "Canned beans": "30.0",
    "Canned tuna": "70.0",
    "Cola": "30.0",
    "Lemon soda": "30.0",
    "Soap": "40.0",
    "Toothpaste": "60.0",
    "Flour": "100.0",
    "Baking powder": "30.0",
    "White bread": "40.0",
    "Whole wheat bread": "50.0",
    "Cornflakes": "100.0",
    "Oats": "100.0",
    "Frozen peas": "100.0",
    "Ice cream": "100.0",
    "Apple": "120.0",
    "Tomato": "70.0",
    "Detergent": "100.0",
    "Dishwashing liquid": "100.0",
    "Prawns": "130.0",
    "Salmon": "170.0",
    "Rice": "80.0",
    "Pasta": "100.0",
    "Batteries": "40.0",
    "Stationary": "30.0"
}

# select product type
product_type = st.selectbox("Select product category", list(product_weights.keys()))

# show products under that product category
products = product_weights[product_type]

product = st.selectbox("Select product", list(products.keys()))

product_weight = products[product]

# product weight text populated from selected product
product_wt = st.number_input("Enter product weight as numeric value in kg", 
                             value=float(product_weight), min_value=0.0, 
                             max_value=1000.0, step=0.01, format="%.2f")

# product ID populated from selected product
product_id = product_codes[product]

prod_id = st.text_input("Product ID", product_id)

# product MRP price populated from selected product
product_MRP_price = product_MRP_prices[product]

prod_MRP_price = st.number_input("Product MRP per kg", 
                                 value=float(product_MRP_price),
                                 min_value=0.0, max_value=1000.0, step=0.01,
                                 format="%.2f")

# list of sugar content options
sugar_options = ["No Sugar", "Low Sugar", "Regular"]

# product sugar content
product_sugar_cont = product_sugar_content[product_type][product]

# Find index of sugar content in sugar options based on product selected
selected_sugar_index = sugar_options.index(product_sugar_cont) if product_sugar_cont in sugar_options else 0

# select right sugar content
prod_sugar_cont = st.selectbox("Product sugar content", sugar_options, index=selected_sugar_index)

# store establishment year
store_establishment_year = st.number_input("Enter 4-digit store establishment year", 
                                           min_value=1970, step=1)

# store type - selection also determines the store size that will be needed
store_type_size_map = {"Departmental Store" : "Medium", "Food Mart" : "Small",
                       "Supermarket Type1" : "High", "Supermarket Type2" : "Medium"}

store_type = st.selectbox("Select store type", list(store_type_size_map.keys()))

store_size_required = store_type_size_map[store_type]

# store size - square footage: High/Medium/Low
store_size_options = ["High", "Medium", "Small"]

# find index associated with required store size based on store type
selected_store_size_index = store_size_options.index(store_size_required) if store_size_required in store_size_options else 0

# display the store size options with preselected size based on chosen store type
store_size = st.selectbox("Select store size based on sq. ft. coverage", store_size_options, index=selected_store_size_index)

# enter store id
store_id = st.text_input("Enter store id", value="STOR01", max_chars=6)

# store_city_type: Tier 1 / Tier 2 / Tier 3
store_city_type = st.selectbox("Select store location city type", ["Tier 1", "Tier 2", "Tier 3"])

# product allocated area: expressed as a fraction of the store capacity
product_allocated_area = st.number_input("Product Allocated Area Ratio between 0 and 0.3", 
                                         min_value=0.004, max_value=0.3, step=0.001, format="%.3f")


def Validate_inputs():
  # validate inputs
  if ((not product_id) or (not re.match('[a-zA-Z0-9]',product_id))):
    st.write("Pls enter Product Id as alphanumeric")
    return False
  elif ((not product_weight) or 
   (not float(product_weight)) or 
    (float(product_weight) <= 0)):
    st.write("Pls enter product weight as any numeric value in kg")
    return False
  elif not product_sugar_cont:
    st.write("Pls select product sugar content")
    return False
  elif ((not store_establishment_year) or 
   (not int(store_establishment_year)) or
       (int(store_establishment_year) >= date.today().year)):
       st.write("Pls enter 4-digit store establishment year less than current year")
       return False
  elif not store_size:
        st.write("Pls select store size")
        return False
  elif not store_city_type:
        st.write("Pls select store location city type")
        return False
  elif not store_type:
        st.write("Pls select store type")
        return False
  elif ((not store_id) or (not re.match('STOR[0-9][0-9]',store_id))):
        st.write("Pls enter store id as format STOR[0-9][0-9]")
        return False
  elif not product_allocated_area:
        st.write("Pls enter product allocated area as a fraction of store space \
         between 0.004 and 0.30 inclusive")
        return False
  else:
    return True

# predict button click
if st.button("Predict"):
  if Validate_inputs():
    # prepare input data for POST request
    input_data = pd.DataFrame({
    "Product_Id": product_id,
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_cont,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_MRP_price,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city_type,
    "Store_Type": store_type
}, index=[0])
    
    st.write("Input data is:\n", input_data)
    
    # apply preprocessing function thru proprocessor joblib
    input_data_processed = preprocessor.transform(input_data)
    
    prediction = model.predict(input_data_processed)
    st.write("Sales Forecast is:\n", prediction)
  else:
    st.write("Invalid inputs: pls correct error")
