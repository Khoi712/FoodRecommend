from flask import Flask, render_template, request, redirect, url_for, session
import random
import secrets
from unidecode import unidecode
import pandas as pd
import numpy as np
from app3 import output_recommendation2, find_store_id
from datetime import datetime, time

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

df_customer = pd.read_excel('data.xlsx', sheet_name='customer')
df_customer['weight'] = pd.to_numeric(df_customer['weight'].astype(str).str.replace('kg', '').str.strip(), errors='coerce')
df_customer['height'] = pd.to_numeric(df_customer['height'].astype(str).str.replace('m', '').str.replace(',', '.'), errors='coerce')

def calculate_bmi(weight, height):
    # Tính BMI
    bmi = weight / ((height / 100) ** 2)  # Giả sử chiều cao đơn vị là centimet
    return bmi

def generate_random_id(existing_ids):
    while True:
        new_id = str(random.randint(10000, 99999))
        if new_id not in existing_ids:
            return new_id

# Route cho trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Route cho trang đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    global df_customer
    if request.method == 'POST':
        user_id = request.form['user_id']
        
        df_customer = pd.read_excel('data.xlsx', sheet_name='customer')
        df_customer['weight'] = pd.to_numeric(df_customer['weight'].astype(str).str.replace('kg', '').str.strip(), errors='coerce')
        df_customer['height'] = pd.to_numeric(df_customer['height'].astype(str).str.replace('m', '').str.replace(',', '.'), errors='coerce')
        
        if user_id.isdigit() and int(user_id) in df_customer['id'].values:
            user_info = df_customer[df_customer['id'] == int(user_id)]
            session['user_id'] = user_id
            return render_template('user_info.html', user_info=user_info)
        else:
            return render_template('login.html', error_message=f"ID {user_id} không tồn tại.")
    return render_template('login.html')

# Route cho trang đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    
    global df_customer  # Use the global keyword to access the global variable

    if request.method == 'POST':
        existing_ids = df_customer['id'].astype(str).tolist()

        id = generate_random_id(existing_ids)
        favorite = request.form['favorite']
        weight = float(request.form['weight'])
        age = int(request.form['age'])
        height = float(request.form['height'])
        gender = request.form['gender']
        bmi = calculate_bmi(weight, height)

        new_data = {
            'id': int(id),
            'favorite': [unidecode(f.strip()) for f in favorite.split(',')],
            'weight': weight,
            'age': age,
            'body fat': f"{bmi:.2f}%",  # Store BMI percentage as a formatted string
            'height': height,
            'gender': gender
        }

        df_customer = pd.concat([df_customer, pd.DataFrame([new_data])], ignore_index=True)
        
        # Read the existing Excel file
        existing_sheets = pd.read_excel('data.xlsx', sheet_name=None)

        # Update the 'customer' sheet
        existing_sheets['customer'] = df_customer

        # Save all sheets back to the 'data.xlsx' file
        with pd.ExcelWriter('data.xlsx', engine='openpyxl') as writer:
            for sheet_name, sheet_df in existing_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

        id = new_data['id']
        return render_template('register_success.html', id=id)

    return render_template('register.html')

@app.route('/order_food', methods=['GET', 'POST'])
def order_food():
    global df_customer
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        delivery_time_str = request.form['delivery_time']
        food_choice_str = request.form['food_choice']
        
        delivery_time = datetime.strptime(delivery_time_str, '%H:%M:%S').time()
        if 'user_id' not in session:
            return redirect(url_for('login'))  # Redirect to login if user_id is not in the session

        # Access user_id from the session
        user_id = int(session['user_id'])
        
        # Tạo một đối tượng Order hoặc thêm dữ liệu vào danh sách order
        order = {
            'delivery_time': delivery_time,
            'food_choice': unidecode(food_choice_str)
        }
        
        input_id = find_store_id(order['food_choice'], 'data.xlsx')
        order['food_choice'] = f"{input_id}_{order['food_choice']}"
        
        print(order['food_choice'])
        option_recommend = output_recommendation2(order)
        
        print(option_recommend)
        return render_template('suggested_food.html', suggested_food = option_recommend)
    
    return render_template('order_food.html')


# @app.route('/suggested_food')
# def suggested_food():
#     # Gọi hàm suggest_food và truyền dữ liệu cho template
#     return render_template('suggested_food.html', suggested_food = option_recommend)


if __name__ == '__main__':
    app.run(debug=True)
