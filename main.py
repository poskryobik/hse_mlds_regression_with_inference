from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import re


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]


# Загрузка inputer
with open("models/imputer.pkl", "rb") as f:
    imputer_1 = pickle.load(f)
# Загрузка poly
with open("models/poly.pkl", "rb") as f:
    poly_1 = pickle.load(f)
# Загрузка ohe_new
with open("models/one_hot.pkl", "rb") as f:
    ohe_new_1 = pickle.load(f)
with open("models/std_scaler.pkl", "rb") as f:
    std_scaler_1 = pickle.load(f)
# Загрузка best_estimator
with open("models/best_estimator.pkl", "rb") as f:
    best_estimator_1 = pickle.load(f)


def change_type(df: pd.DataFrame):
    """ 
        Функция для исключения единиц измерения 
    """
    columns = ['mileage', 'engine', 'max_power']
    for column in columns:
        # Замена пустых значений пустой строкой
        df[column] = df[column].fillna('')
        # Регуляркой откидываются ненужные символы
        df[column] = df[column].apply(lambda x: re.sub(r'[^0-9.]*', '', x))
        df[column] = pd.to_numeric(df[column], downcast='float')
    return df.copy()

def get_poly_feature(df, poly):
    """ 
        Добавление к имеющемуся датафрейму полиномиальных признаком 
        для числовых признаков, кроме seat
    """
    poly_df = pd.DataFrame(poly.transform(df[['year', 'km_driven', 'mileage', 'engine', 'max_power']]),
                columns=poly.get_feature_names_out())
    df = df.drop(['year', 'km_driven', 'mileage', 'engine', 'max_power'],axis=1)
    df = pd.concat([df, poly_df], axis=1)
    return df

def get_one_hot(df, ohe, cat_columns: list, drop=True):
    """ 
        Функция добавляет one_hot значение 
        категориальных фичей имеющегося датасета
    """
    # Полученные данные 
    ohe_data = (
        pd.DataFrame(
            ohe.transform(df[cat_columns]).toarray(), 
            columns = ohe.get_feature_names_out()))
    if drop:
        df = df.drop(cat_columns, axis=1)
    # Формирование тренироваочного датасета с one_hot значениями
    df = pd.concat([df, ohe_data], axis=1)
    return df

def preprocessing_data(df, imputer, poly, one_hot, std_scaler):
    """ 
        Функиця предобработки входных данных
    """
    # 1 Убираем единицы измерения 
    df = change_type(df=df)
    # 2 Удаление torque
    df = df.drop('torque', axis=1)
    # 3 Достаем марку машины
    df['brand'] = df['name'].apply(lambda x: re.findall(r"\b\w*", x)[0])
    # 4 удаление имени
    df = df.drop(['name'], axis=1)
    # 5 Замена пустых значений 
    columns_for_input = ['mileage', 'engine', 'max_power'] 
    df[columns_for_input] = (
        pd.DataFrame(imputer.transform(df[columns_for_input]), 
                    columns=columns_for_input))
    # 6 Добавление полиномиальных параметров
    df = get_poly_feature(df=df, poly=poly)

    # 7 One_hot кодирование
    for col in ['brand', 'seats']:
        df[col] = df[col].astype('category')
        df[col] = df[col].astype('category')   
    # Категориальные колонки
    cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brand']
    df = get_one_hot(df=df, ohe=one_hot, 
                cat_columns=cat_columns, drop=True)
    # 8 Стандартизация признаков
    columns_for_std = ['1', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'year^2',
        'year km_driven', 'year mileage', 'year engine', 'year max_power',
        'km_driven^2', 'km_driven mileage', 'km_driven engine',
        'km_driven max_power', 'mileage^2', 'mileage engine',
        'mileage max_power', 'engine^2', 'engine max_power', 'max_power^2']
    df[columns_for_std] = pd.DataFrame(std_scaler.transform(df[columns_for_std]), columns=columns_for_std)
    # Выделение таргета
    y_true = df['selling_price']
    df = df.drop('selling_price', axis=1)
    return y_true, df  




@app.post("/uploadcsv")
async def create_upload_file(file: UploadFile):
    """ 
        на вход подается csv-файл с признаками тестовых объектов, 
        на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах
    """
    csv_df = pd.read_csv(file.file)
    y_true, x_input = preprocessing_data(df=csv_df, imputer=imputer_1, poly=poly_1,
                                     one_hot=ohe_new_1, std_scaler=std_scaler_1) 
    csv_df['predict'] = best_estimator_1.predict(X=x_input)
    csv_df.to_csv("result.csv")
    return FileResponse("result.csv", media_type="text/csv", filename=file.filename)


@app.get("/getcsv")
def create_upload_file():
    return FileResponse("result.csv", media_type="text/csv", filename='result.csv')


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """ 
        метод post, который получает на вход один объект описанного класса Item
        возвращает predict: float
    """
    # Формирование датафрейма из входного json
    input_df = pd.DataFrame.from_dict(pd.json_normalize(item.dict()))
    y_true, x_input = preprocessing_data(df=input_df, imputer=imputer_1, poly=poly_1,
                                     one_hot=ohe_new_1, std_scaler=std_scaler_1) 
    answer = best_estimator_1.predict(X=x_input)[0]
    return answer

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    """ 
        метод post, который получает на вход коллекцию (list) объектов описанного класса Item
        возвращает список предиктов List[float]
    """
    # Приведение типов экземпляра класса к словарю
    for i, item in enumerate(items):
        items[i] = item.dict()
    input_df = pd.DataFrame.from_dict(pd.json_normalize(items))
    y_true, x_input = preprocessing_data(df=input_df, imputer=imputer_1, poly=poly_1,
                                     one_hot=ohe_new_1, std_scaler=std_scaler_1) 
    input_df['predict'] = best_estimator_1.predict(X=x_input)
    return input_df['predict'].to_list()
