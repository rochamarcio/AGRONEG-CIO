import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings ("ignore")



def preprocessing():

   train_d = pd.read_csv("../../Dados/treino.csv")

   train_d.drop ('ID', axis=1,inplace=True)
   train_d.head()

   train_d.dropna(inplace=True)

   for col in ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season']:
    train_d = pd.get_dummies(train_d,columns=[col])
    
   X = train_d.drop(['Crop_Damage'],axis=1)
   y = train_d['Crop_Damage'].values.reshape(-1,1)

   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7)
   return X_train, X_test, y_train, y_test


#if __name__ == "__main__":
    # Chama sua função principal ou outras funções conforme necessário
 #   resultado = sua_funcao_principal()
  #  print("Resultado:", resultado)
