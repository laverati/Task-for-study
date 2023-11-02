import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # Pipeline.Не добавить, не убавить

from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.compose import TransformedTargetRegressor

from sklearn.preprocessing import PowerTransformer  # Степенное преобразование от scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #

from sklearn.base import BaseEstimator, TransformerMixin # для создания собственных преобразователей / трансформеров данных

from sklearn.linear_model import SGDRegressor 


from grafs.first import get_fisrt_dfgdfg
from sheet.pipeline import num_pipes, cat_pipes

def cat_and_nums(df): #Категориальные и числовые признаки
    cat_columns = [] # создаем пустой список для имен колонок категориальных данных
    num_columns = [] # создаем пустой список для имен колонок числовых данных

    for column_name in df.columns: # смотрим на все колонки в датафрейме
        if (df[column_name].dtypes == object): # проверяем тип данных для каждой колонки
            cat_columns +=[column_name] # если тип объект - то складываем в категориальные данные
        else:
            num_columns +=[column_name] # иначе - числовые
    
    return cat_columns, num_columns[1:]

df = pd.read_csv('./datasets/studentInfo.csv')  

## Получение базовой инфы

# print(df.info())
# print(df.describe())


num_columns = cat_and_nums(df)[1]
cat_columns = cat_and_nums(df)[0]


# get_fisrt_dfgdfg(num_columns, df)

# ordinal = OrdinalEncoder()
# ordinal.fit(df[cat_columns])
# Ordinal_encoded = ordinal.transform(df[cat_columns])
# df_ordinal = pd.DataFrame(Ordinal_encoded, columns = cat_columns)
# print(df_ordinal)


# не забываем удалить целевую переменную цену из признаков
X,y = df.drop(columns = ['final_result']), df['final_result']

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=0.8,
                                                    random_state=42)


preprocessors_num = ColumnTransformer(transformers=num_pipes(num_columns))
all_list = num_pipes(num_columns) + cat_pipes(cat_columns)

# и Pipeline со всеми признаками
preprocessors_all = ColumnTransformer(transformers=all_list)

pipe_all_transform = Pipeline([
    ('preprocessors', preprocessors_all),
    ('model', TransformedTargetRegressor( regressor=SGDRegressor(random_state = 42),
    transformer=StandardScaler())
    )
])

Label = LabelEncoder()
Label.fit(y) # задаем столбец, который хотим преобразовать
Label.classes_ 