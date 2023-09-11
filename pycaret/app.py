import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import streamlit as st 

from sklearn.model_selection import train_test_split
from EDA import EDA,plotting
from ml import Auto_ML
def get_data(file):   
    try:
        df = pd.read_excel(file)
    except:
        try:
              
              df=pd.read_csv(file)
        except:      
              df=pd.DataFrame()
    return df

def get_info(df):
    return pd.DataFrame({'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum()/len(df))*100,2), 'unique':df.nunique()})
def get_stats_num(df):
        return df.describe().transpose()

def get_state_cat(df):
     df_cat = df.select_dtypes(include = 'object')
     return df_cat.describe().transpose()

def info(df):
    st.header("about yor data")
    st.write('Number of observations', df.shape[0]) 
    st.write('Number of variables', df.shape[1])
    st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))

    st.markdown('**Numerical summary**')
    df_num = df.select_dtypes(include = np.number)
    if df_num.shape[1] >=1:
        df_num = df.select_dtypes(include = np.number)
        state_num = get_stats_num(df_num)
        st.table(state_num)
    else :
        st.write("there's no numeriacl feature")
    st.markdown('**Categorical summary**')
    df_cat = df.select_dtypes(exclude = np.number)
    if df_cat.shape[1] >=1:
        state_cat = get_state_cat(df_cat)
        st.table(state_cat)
    else:
        st.write("there's no categorical feature")
    st.markdown('**Missing Values**')
    df_info = get_info(df)
    st.table(df_info)

     
def main ():
    st.title('Automated ML Models')
    st.write("AutoML can automatically handle various preprocessing steps and model training, making it easy to experiment with different algorithms and hyperparameters.")
    file  = st.file_uploader('Upload your file ', type = ['csv','xlsx'])
    if file is not None:
        df = get_data(file)
        show = st.checkbox('show information about your data')
        if show:
            info(df)
        st.header('... Model Testing ...')
        st.markdown("show every model training - testing score with best hyperparamter")
        target = st.selectbox('chose your target variable',  df.columns.insert(0,None))
        if target != None:
            X = df.drop(target, axis = 1)
            y = df[target]
            eda = EDA(X,y)
            X,y  = eda.Wrangle()
            cols = X.columns
            X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42, shuffle= True)
            auto_ml = Auto_ML(X_train, X_test, y_train, y_test,cols)
            if df[target].dtypes in ['int64','float'] and df[target].nunique()>10:
                st.write('your target indicate to Regression problem')
                
                Rmodels= st.radio('Regression Models', ('None','LinearRegression','Ridge', 'SVR', 'DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','GradientBoostingRegressor'))
                if Rmodels =='LinearRegression':
                    lr,report= auto_ml.Linear_regression()
                    st.write(lr)
                    st.table(report)
                if Rmodels == 'Ridge':
                    ridge ,report= auto_ml.ridge()
                    st.write(ridge)
                    st.table(report)
                if Rmodels == 'SVR':
                    svr,report = auto_ml.support_vector_regressor()
                    st.write(svr)
                    st.table(report)
                if Rmodels == 'DecisionTreeRegressor':
                    dtr,report = auto_ml.DTR()
                    st.write(dtr)
                    st.table(report)
                if Rmodels == 'KNeighborsRegressor':
                    knnr,report = auto_ml.KNNR()
                    st.write(knnr)
                    st.table(report)
                if Rmodels == 'RandomForestRegressor':
                    rfr,report = auto_ml.RFR()
                    st.write(rfr)
                    st.table(report)
                if Rmodels == 'GradientBoostingRegressor':
                    gbr,report = auto_ml.GBR()
                    st.write(gbr)
                    st.table(report)
            else :
                st.write('your target indicate to classification problem')
                Cmodels= st.radio('classification Models', ('None','LogisticRegression', 'SVC', 'DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier','GradientBoostingClassifier'))

                if Cmodels =='LogisticRegression':
                    log_r,report,fig,fig2= auto_ml.log_regression()
                    st.write(log_r)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)
                    
                if Cmodels == 'SVC':
                    svc,report,fig,fig2 = auto_ml.svc()
                    st.write(svc)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)
                if Cmodels == 'DecisionTreeClassifier':
                    dtc,report,fig,fig2 = auto_ml.DTC()
                    st.write(dtc)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)
                if Cmodels == 'RandomForestClassifier':
                    rfc,report,fig ,fig2= auto_ml.RFC()
                    st.write(rfc)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)
                if Cmodels == 'KNeighborsClassifier':
                    knnc,report,fig,fig2 = auto_ml.KNNC()
                    st.write(knnc)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)
                if Cmodels == 'GradientBoostingClassifier':
                    gbc,report,fig ,fig2= auto_ml.GBC()
                    st.write(gbc)
                    st.table(report)
                    st.pyplot(fig)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(fig2)

       
if __name__ == '__main__':
    main()




