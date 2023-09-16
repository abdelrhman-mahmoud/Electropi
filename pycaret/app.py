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

        eda = EDA(df)
        st.header("Data Preprocessing")
        st.write('.. starting with numerical features ..')

        cols = eda.nums_nulls()
        
        if len(cols) !=0:
            for i, col in enumerate(cols):
                st.write(col)
                tech = st.selectbox('select technique',('mean','median','mode'),key = f'A{i}')
                if tech == 'mean':
                    df = eda.fill_with_mean(col)
                
                if tech == 'median':
                    df = eda.fill_with_median(col)
                    
                if tech == 'mode':
                    df = eda.fill_with_mode(col)
        else :
            st.write('no detected nulls in numerical features')

        st.write(" ")
        st.write( 'following with categorical features')
        cat_cols = eda.cat_nulls()
        if len(cat_cols) !=0:
            for i, col in enumerate(cat_cols):
                tech = st.selectbox('select technique',('mode','Other'),key = f'B{i}')
                if tech == 'mode':
                    df = eda.fill_with_mode(col)
                    
                if tech == 'Other':
                    value = st.text_input('Enter your constant value')
                    df = eda.fill_with_constant(col,value)
                    
 
        else :
            st.write('no detected nulls in categorical features')

        st.write(" ")

        st.write("Great, You have completed the first step of data preprocessing")
        st.write(" ")
        st.write(" ")
        # st.write("it's time to encode your categorical feature")
    
        st.write('select the target variable to auto detect problem(Regression - Classification)')
        target = st.selectbox('choose your target variable',  df.columns.insert(0,'None'))
      

                
        if target != 'None':
            problem = None
            if df[target].dtypes in ['int64','float'] and df[target].nunique()>=10:
                problem = 'Regression'
            else :
                problem = 'Classification'
                df = eda.label_encoder(target)
            df = eda.encoding(df)
            X = df.drop(target, axis = 1)
            y = df[target]
            eda = EDA(X)
            cat_cols = eda.cat_feat()
            
            X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42, shuffle= True)
            cols  = X.columns
            auto_ml = Auto_ML(X,y,X_train, X_test, y_train, y_test,cols)
           
            if problem == 'Regression':
                st.write('your target indicate to Regression Problem')
                st.header('Model Testing ')
                st.write('Select a Regression Model to show evalution with hyperparameter tuning ')
                Rmodels= st.radio('Regression Models', ('None','LinearRegression','Ridge', 'SVR', 'DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','GradientBoostingRegressor'))
                if Rmodels =='LinearRegression':
                    
                    model,report,graph1,graph2= auto_ml.Linear_regression()
                    st.write(model)
                    st.table(report)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)
                   
            
                if Rmodels == 'Ridge':
                    model,report,graph1,graph2 = auto_ml.ridge()
                    st.write(model)
                    st.table(report)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)
                
                if Rmodels == 'SVR':
                    model,report,fig= auto_ml.support_vector_regressor()
                    st.write(model)
                    st.table(report)
                    
                    st.pyplot(fig)

                if Rmodels == 'DecisionTreeRegressor':
                    model,report,graph1, graph2= auto_ml.DTR()
                    st.write(model)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)

                if Rmodels == 'KNeighborsRegressor':
                    model,report,graph1,graph2= auto_ml.KNNR()
                    st.write(model)
                    st.table(report)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)

                if Rmodels == 'RandomForestRegressor':
                    model,report,graph1, graph2= auto_ml.RFR()
                    st.write(model)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)
                if Rmodels == 'GradientBoostingRegressor':
                    model,report,graph1, graph2= auto_ml.GBR()
                    st.write(model)
                    st.pyplot(graph1)
                    plt.close(graph1)
                    st.pyplot(graph2)
                    plt.close(graph2)

            else :
                st.write('your target indicate to Classification Problem')
                st.header('Model Testing ')
                st.write('Select a Classification Model to show evalution with hyperparameter tuning ')
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




