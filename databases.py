import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



class Database(object):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

def get_db(name):
    if name == "life expectancy":
        return create_life_expectancy_database()
    elif name == "diamonds":
        return create_diamonds_database()
    elif name == "car prices":
        return create_car_prices_database()

def create_life_expectancy_database():
    raw_data = pd.read_csv('resources/Life Expectancy Data.csv')

    # print(((raw_data.isna().sum() / 3938)*100))

    raw_data = raw_data.fillna(raw_data.mean().iloc[0])

    numeric_cols = raw_data[["Adult Mortality",	"infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
                 " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ", " HIV/AIDS", "GDP",
                 "Population", " thinness  1-19 years", " thinness 5-9 years", "Income composition of resources", "Schooling", "Life expectancy "]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df["Country"] = raw_data["Country"]
    df["Year"] = raw_data["Year"]
    df["Status"] = raw_data["Status"]

    encoder = LabelEncoder()
    df['Country'] = encoder.fit_transform(df['Country'])
    df['Status'] = encoder.fit_transform(df['Status'])
    df['Year'] = encoder.fit_transform(df['Year'])

    # df = pd.get_dummies(df, columns=['Country', 'Year', 'Status'])

    y = np.array(df['Life expectancy '])

    x = df.drop(columns="Life expectancy ")

    life_expectancy_db = Database(x, y)

    return life_expectancy_db

def create_diamonds_database():
    data = pd.read_csv('resources/diamonds.csv')
    data = data.drop(columns=['Unnamed: 0'])
    encoder = LabelEncoder()
    data['cut'] = encoder.fit_transform(data['cut'])
    data['color'] = encoder.fit_transform(data['color'])
    data['clarity'] = encoder.fit_transform(data['clarity'])
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']].replace(0, np.NaN)
    data = data.dropna(how='any')
    # print(data.isnull().sum())

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    x = data.drop(['price'], 1)
    y = data['price']



    return Database(x, y)

def create_car_prices_database():
    raw_data = pd.read_csv('resources/car_train.csv')
    raw_data = raw_data.drop(columns=['rownum','acquisition_date','last_updated'])
    raw_data = raw_data.dropna(subset=['price', 'body_type', 'fuel', 'transmission'])
    raw_data = raw_data.dropna()

    # raw_data['economy'].fillna(raw_data['economy'].mean(), inplace=True)
    # raw_data['odometer'].fillna(raw_data['odometer'].mean(), inplace=True)
    # raw_data['badge'].fillna(raw_data['badge'].mode()[0], inplace=True)
    # raw_data['category'].fillna(raw_data['category'].mode()[0], inplace=True)
    # raw_data['colour'].fillna(raw_data['colour'].mode()[0], inplace=True)
    # raw_data['cylinders'].fillna(raw_data['cylinders'].mode()[0], inplace=True)
    # raw_data['litres'].fillna(raw_data['litres'].mode()[0], inplace=True)

    numeric_cols = raw_data[["odometer","price"]]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df['badge'] = raw_data['badge']
    df['body_type'] = raw_data['body_type']
    df['category'] = raw_data['category']
    df['colour'] = raw_data['colour']
    df['cylinders'] = raw_data['cylinders']
    df['economy'] = raw_data['economy']
    df['fuel'] = raw_data['fuel']
    df['litres'] = raw_data['litres']
    df['location'] = raw_data['location']
    df['make'] = raw_data['make']
    df['model'] = raw_data['model']
    df['transmission'] = raw_data['transmission']
    df['year'] = raw_data['year']
    # df['price'] = raw_data['price']

    encoder = LabelEncoder()
    df['badge'] = encoder.fit_transform(df['badge'])
    df['body_type'] = encoder.fit_transform(df['body_type'])
    df['category'] = encoder.fit_transform(df['category'])
    df['colour'] = encoder.fit_transform(df['colour'])
    df['cylinders'] = encoder.fit_transform(df['cylinders'])
    df['economy'] = encoder.fit_transform(df['economy'])
    df['fuel'] = encoder.fit_transform(df['fuel'])
    df['litres'] = encoder.fit_transform(df['litres'])
    df['make'] = encoder.fit_transform(df['make'])
    df['model'] = encoder.fit_transform(df['model'])
    df['transmission'] = encoder.fit_transform(df['transmission'])
    df['year'] = encoder.fit_transform(df['year'])

    y = np.array(df['price'])
    x = df.drop(columns=['price'])
    car_price_db = Database(x, y)

    return car_price_db