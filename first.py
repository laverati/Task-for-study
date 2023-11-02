import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

def get_fisrt_dfgdfg(num_columns, df):
    
    width = 2
    height = int(np.ceil(len(num_columns)/width))
    fig, ax = plt.subplots(nrows=height, ncols=width, figsize=(16,8)) 
    

    for idx, column_name in enumerate(num_columns): # перебираем все числовые данные
        plt.subplot(height,width, idx+1) #берем конкретную ячейку из заранее подготовленную заготовку
        # рисуем с помощью библиотеки seaborn
        sns.histplot(data=df, # какой датафрейм используем
                x=column_name, # какую переменную отрисовываем
                bins = 20);  # на сколько ячеек разбиваем

    plt.savefig("first.png")

    return True
