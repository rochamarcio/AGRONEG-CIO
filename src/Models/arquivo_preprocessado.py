import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings ("ignore")

# Criar uma função de imputação
def imputar(df):
    if df['Number_Weeks_Used'].isnull().any():
        df['Number_Weeks_Used'] = df['Number_Weeks_Used'].fillna(df['Number_Weeks_Used'].mean())
    return df

def preprocessing():

   train_d = pd.read_csv("../../Dados/treino.csv")
   
# Aplicar a função de imputação usando apply
   df = train_d.groupby('Crop_Damage', group_keys=False).apply(imputar)

   df.drop ('ID', axis=1,inplace=True)

   df.dropna(inplace=True)

   for col in ['Crop_Type','Soil_Type','Pesticide_Use_Category','Season']:
    df = pd.get_dummies(df,columns=[col])
    
   X = df.drop(['Crop_Damage'],axis=1)
   y = df['Crop_Damage'].values.reshape(-1,1)

   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=20, stratify=y)
   return X_train, X_test, y_train, y_test


#if __name__ == "__main__":
    # Chama sua função principal ou outras funções conforme necessário
 #   resultado = sua_funcao_principal()
  #  print("Resultado:", resultado)
