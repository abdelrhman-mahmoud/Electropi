import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
from sklearn.impute import SimpleImputer
from sklearn. preprocessing import LabelEncoder,OneHotEncoder
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


class EDA:

    def __init__(self,df):
        self.df = df 
    
    def cat_feat(self):
        df_cat = self.df.select_dtypes(include ='object')
        cols = df_cat.columns
        return cols
     
        
    def cat_nulls(self):
        df_cat = self.df.select_dtypes(include ='object')
        x = df_cat.columns[df_cat.isna().sum()>0]
        cols = []
        for i in x:
            cols.append(i)
        return cols
    
    def nums_nulls(self):
        df_num = self.df.select_dtypes(include = np.number)
        x = df_num.columns[df_num.isna().sum()>0]
        cols = []
        for i in x:
            cols.append(i)
        return cols


    def fill_with_mean(self,col):
        impute_mean = SimpleImputer(strategy='mean')
        self.df[col]= impute_mean.fit_transform(self.df[col].values.reshape(-1,1))
        return self.df
    
    def fill_with_median(self,col):
        impute_median = SimpleImputer(strategy='median')
        self.df[col]= impute_median.fit_transform(self.df[col].values.reshape(-1,1))
        return self.df
    
    def fill_with_mode(self,col):
        most_frequent_value = self.df[col].mode()[0]
        self.df[col].fillna(most_frequent_value, inplace=True)
        return self.df

    def fill_with_constant(self,col,value):
        self.df[col].fillna(value, inplace=True)
        return self.df
    
    
    def one_hot_encoder(self,col):
        OHE = OneHotEncoder( sparse =False, drop='first')
        df_encoded = OHE.fit_transform(self.df[col].values.reshape(-1,1))
        encoded_feature_names = self.df[col].unique()[1:]
        encoded_df = pd.DataFrame(df_encoded, columns=encoded_feature_names)
        df = self.df.drop(col,axis = 1)
        df = pd.concat([df,encoded_df],axis = 1)
        return df
    
    def label_encoder(self,col):
        LE = LabelEncoder()
        self.df[col] = LE.fit_transform(self.df[col])
        return self.df
    
    def encoding(self,df):
        df1 = pd.get_dummies(df,drop_first= True)
        return df1



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


    # Create a function to plot the learning curve
    def plot_learning_curve(self,estimator, title, X, y):

        fig, ax = plt.subplots()
       
        plt.title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("R2 Score")  # You can use other regression metrics as needed
        ax.set_title(title)

        train_sizes, train_scores,test_scores = learning_curve(
            estimator, X, y, cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0), n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring='r2')
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        ax.legend(loc="best")
        return fig
        
    
    def feature_importance_plot(self,feature_importances,feature_names):
        fig, ax = plt.subplots()

        feat_imp = pd.Series(feature_importances, index =feature_names ).sort_values().tail().plot(kind = 'barh',xlabel = 'Importance', ylabel='features',title ='Top important features')
        return fig

  