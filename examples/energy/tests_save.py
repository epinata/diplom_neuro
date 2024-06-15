import NFuzMatrix # нейронная сеть
from NFuzMatrix import NFM # нейронная сеть
import os # работа с файлами
import numpy as np
import pandas as pd
import pickle  # сохрание и загрузка состояния нейросети 

# загрузка состояния сети
with open('NeuFuzMatrix_model.pkl', 'rb') as f:
    nfm_loaded = pickle.load(f)
# Текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к файлу относительно текущей директории
df_train = pd.read_csv("Test.csv")
X = np.array(df_train[["AT", "V"]])
y_test = NFM.predict(nfm_loaded, X)
print(y_test)