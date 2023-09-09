import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.over_sampling import SMOTE

class EDA:
    def __init__(self,df,y):
        self.df = df
        self.y = y
    def Wrangle(self):
        # Drop columns that are more than 50% null values.
        thresh = self.df.shape[0]//2
        self.df.dropna(inplace = True,thresh = thresh, axis = 1)
        
        df_number = self.df.select_dtypes(include = np.number)
        df_cat = self.df.select_dtypes(include = 'object')
        
        # drop cols that have high/low cardinality 
        LH_cardinality = [col for col in df_cat.columns if(df_cat[col].nunique() < 2 or df_cat[col].nunique() >10)]
        df_cat.drop(LH_cardinality,axis =1,inplace = True)
        
        for col in df_cat.columns:
            most_frequent_value = df_cat[col].mode()[0]
            df_cat[col].fillna(most_frequent_value, inplace=True)
        
        df_number.fillna(df_number.mean(),inplace = True)
        
        median = np.nanmedian(self.y)
        y = np.where(np.isnan(self.y), median, self.y)
        
        df1 = pd.concat([df_number, df_cat], axis = 1)
        #encoding categorical features
        df1 = pd.get_dummies(df1,drop_first= True)
        
        return df1,y
    
        