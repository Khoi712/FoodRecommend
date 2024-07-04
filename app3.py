
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
import string
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from datetime import datetime, time

food_menu = pd.DataFrame(pd.read_excel('data.xlsx', sheet_name='food_menu'))
store = pd.DataFrame(pd.read_excel('data.xlsx', sheet_name='store'))

"""# **STORE**

*   Input: category
*   Output: store name with most similar
*   Based on: description of food / drink
"""

def cleaning_distance(text):
    match = re.sub(r'km', '', text)
    if match:
        return float(match)
    else:
        return 0
store['distance'] = store['distance'].apply(cleaning_distance)

store2 = store

features = ['store_id','distance', 'rating', 'name', 'description', 'category', 'discount_percent', 'discount_time']

store2 = store2[features]

def create_soup(x):
    return str(x['category']) + " " + str(x['name']) + " " + str(x['description']) + " " + str(x['distance']) + " " + str(x['rating']) + " " + str(x['discount_percent']) + " " + str(x['discount_time'])
store2['soup'] = store2.apply(create_soup, axis=1)

# Đọc file các từ dừng (stopwords) những từ không có tác dụng trong việc so sánh, đánh giá
vietnamese_stop_words_path = 'vietnamese-stopwords.txt'

with open(vietnamese_stop_words_path, 'r', encoding='utf-8') as file:
    vietnamese_stop_words = [line.strip() for line in file]

count = CountVectorizer(stop_words = vietnamese_stop_words)
count_matrix = count.fit_transform(store2['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

store2 = store2.reset_index()
indices = pd.Series(store2.index, index = store2['store_id'])

def get_recommendations_store(title, cosine_sim=cosine_sim):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar food
    sim_scores = sim_scores[1:10]
    store_indices = [i[0] for i in sim_scores]

    recommendations = store2.iloc[store_indices][['store_id','name', 'rating','distance','discount_percent']]
    return recommendations

recommendations = get_recommendations_store('10T')
recommendations.to_csv('recommendations.csv', index=False)

store3=store

"""# **FOOD MENU**

**Simple Content Based Filtering**

*   Input: Food / Drink name
*   Output: Food / Drink name with most similar
*   Based on: description of food / drink
"""

def cleaning_calo(text):
    match = re.search(r'\b\d+\b', text)
    if match:
        return int(match.group())
    else:
        return 0

def cleaning_price(text):
    match = re.sub(r'k', '', text)
    if match:
        return int(match)
    else:
        return 0
food_menu['calo'] = food_menu['calo'].apply(cleaning_calo)
food_menu['price'] = food_menu['price'].apply(cleaning_price)

food_menu_2 = food_menu

# Loại bỏ các dấu chấm câu
def text_cleaning(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    return text

food_menu_2['description'] = food_menu_2['description'].apply(text_cleaning)

from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc file các từ dừng (stopwords) những từ không có tác dụng trong việc so sánh, đánh giá
vietnamese_stop_words_path = 'vietnamese-stopwords.txt'

with open(vietnamese_stop_words_path, 'r', encoding='utf-8') as file:
    vietnamese_stop_words = [line.strip() for line in file]

# Tạo index bằng cách kết hợp 'store_id' và 'food_name'
# Nếu food_name rỗng (thực phẩm không phải món ăn mà là đồ uống), thì  kết hợp 'store_id' và 'drink_name'
def name(food_menu_2):
    if pd.notna(food_menu_2['food_name']):
        return food_menu_2['food_name']
    elif pd.notna(food_menu_2['drink_name']):
        return food_menu_2['drink_name']
    else:
        return None

food_menu_2['name'] = food_menu_2.apply(name, axis=1)

tfidf = TfidfVectorizer(stop_words=vietnamese_stop_words)
tfidf_matrix = tfidf.fit_transform(food_menu_2['description'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim

indices = pd.Series(food_menu_2.index, index=food_menu_2['name'])
indices

def get_recommendations_store(title, cosine_sim=cosine_sim):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar food
    sim_scores = sim_scores[1:10]
    store_indices = [i[0] for i in sim_scores]

    recommendations = store2.iloc[store_indices][['store_id','food_name', 'drink_name','price','calo']]
    return recommendations

features = ['food_name','types_food', 'description']

def create_soup(x):
    return str(x['food_name']) + " " + str(x['types_food']) + " " + str(x['description'])
food_menu['soup'] = food_menu.apply(create_soup, axis=1)

count = CountVectorizer(stop_words = vietnamese_stop_words)
count_matrix = count.fit_transform(food_menu['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

def create_new_index(row):
    if pd.notna(row['food_name']) and pd.notna(row['store_id']):
        return f"{row['store_id']}_{row['food_name']}"
    else:
        return None
food_menu['new_index'] = food_menu.apply(create_new_index, axis=1)

food_menu = food_menu.reset_index()
indices = pd.Series(food_menu.index, index = food_menu['new_index'])

def get_recommendations2(title, cosine_sim=cosine_sim2):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar food
    sim_scores = sim_scores[1:10]
    food_indices = [i[0] for i in sim_scores]

    recommendations = food_menu.iloc[food_indices][['store_id','food_name','price','calo']]
    return recommendations

def find_store_id(food_name_input, csv_path):
    df = pd.read_excel(csv_path, sheet_name='food_menu')

    # Convert về chữ thường
    food_name_input = food_name_input.lower()

    # Tìm food_name_input trong df['food_name']
    mask = df['food_name'].str.lower() == food_name_input

    # Lấy store_id đầu tiên
    first_store_id = df.loc[mask, 'store_id'].iloc[0] if mask.any() else None

    return first_store_id

def check_discount(row, time_only):
    start_discount = row['start_discount'] * 60
    end_discount = row['end_discount'] * 60
    time_only_minutes = time_only.hour * 60 + time_only.minute  # Convert time_only to minutes

    if start_discount <= time_only_minutes <= end_discount:
        return row['discount_percent'] / 100
    else:
        return 0

def extract_discount_info(df):
    # Define a regex pattern to extract discount percentage, start time, and end time
    pattern = r'giảm (\d+)% từ (\d+)h-(\d+)h'

    # Filter rows where 'uu_dai' matches the pattern
    mask = df['uu_dai'].str.contains(pattern, na=False)  # na=False to treat NaN as not matching
    df = df[mask]

    # Extract discount information using the regex pattern
    discount_info = df['uu_dai'].str.extract(pattern, expand=True)

    # Set default values for rows where the pattern doesn't match
    default_values = {'0': 0, '1': 0, '2': 0}
    discount_info = discount_info.fillna(default_values)

    # Convert columns to appropriate data types
    discount_info[0] = discount_info[0].astype(float) / 100  # Convert to decimal
    discount_info[1] = discount_info[1].astype(int)
    discount_info[2] = discount_info[2].astype(int)

    # Rename columns
    discount_info.columns = ['discount_percent', 'start_discount', 'end_discount']

    # Concatenate the original DataFrame with the extracted discount information
    df = pd.concat([df, discount_info], axis=1)

    return df



def output_recommendation2( order):


    output_df = get_recommendations2(order['food_choice'])


    # Thay đổi tên file Excel và tên sheet theo dữ liệu thực tế của bạn
    excel_file_path = 'data.xlsx'
    store_sheet_name = 'store'
    food_menu_sheet_name = 'food_menu'
    customer_sheet_name = 'customer'

    # Tạo một dataframe mẫu
    temp_df = output_df
    # Đọc dữ liệu từ sheet "store"
    df_store = pd.read_excel(excel_file_path, sheet_name=store_sheet_name)

    # Đọc dữ liệu từ sheet "food_menu"
    df_food_menu = pd.read_excel(excel_file_path, sheet_name=food_menu_sheet_name)

    # Đọc dữ liệu từ sheet "customer"
    df_customer = pd.read_excel(excel_file_path, sheet_name=customer_sheet_name)

    # Kết hợp dữ liệu từ "store" vào "food_order" dựa trên "store_id"
    temp_df = pd.merge(temp_df, df_store[['store_id', 'name', 'rating','uu_dai']], on='store_id', how='left')
    temp_df

    temp_df = extract_discount_info(temp_df)
    # Apply the check_discount function to each row of temp_df and store the result in a new column 'discount_applied'
    temp_df['discount_applied'] = temp_df.apply(check_discount, axis=1, time_only=order['delivery_time'])

    # Selecting columns from temp_df
    selected_columns = ['store_id', 'name', 'rating', 'food_name', 'price', 'calo', 'discount_percent']

    result = pd.DataFrame()

    result[selected_columns] = temp_df[selected_columns]

    return result
