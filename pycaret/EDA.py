import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 

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


class plotting:
    def __init__(self):
        pass
    def plot_confusion_matrix(self,confusion_matrix, labels):
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=labels, yticklabels=labels,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")
        return fig
    
    def bar_plot(self, feat_imp):
        return feat_imp.sort_values(key = abs).tail(10).plot(kind = 'barh',xlabel = 'Importance', ylabel='features',title ='Top 10 important features')
    
    def ORC_plot(self,fpr, tpr,roc_auc):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        