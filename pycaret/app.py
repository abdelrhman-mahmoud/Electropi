import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import streamlit as st 

from sklearn.model_selection import train_test_split
from EDA import EDA,plotting
from ml import Auto_ML
from pycaret_ml import reg, classification       



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
        colums_to_drop = st.multiselect('choose columns to drop :',df.columns)
        
        if colums_to_drop :
            df = df.drop(colums_to_drop,axis = 1)
            eda = EDA(df)
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
        st.write("Your Data after Preprocessing")
        st.write(df)
        # st.write("it's time to encode your categorical feature")
        st.write('Select the target to detect problrm (Regression - Classification)')
        target = st.selectbox('chose your target variable',  df.columns.insert(0,'None'))
      

                
        if target != 'None':
            problem = None
            if df[target].dtypes in ['int64','float'] and df[target].nunique()>=10:
                problem = 'Regression'
            else :
                problem = 'Classification'
                df = eda.label_encoder(target)
                
                
                
            my_report = st.checkbox('My Report')
            if my_report:
            
                X = df.drop(target, axis = 1)
                X = eda.encoding(X)
                y = df[target]
                eda = EDA(X)
                df = pd.concat([X,y],axis = 1)
                show = st.checkbox('show data after encoding')
                if show :
                    st.write(df)

                X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42, shuffle= True)
                cols  = X.columns
                auto_ml = Auto_ML(X,y,X_train, X_test, y_train, y_test,cols)

                if problem == 'Regression':
                    st.write('your target indicate to Regression Problem')
                    st.header('Model Testing')
                    st.write('Select a Regression Model to show evalution of model with best hyperparamter that fit your data')
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
                    st.header('Model Testing')
                    st.write('Select a Regression Model to show evalution of m odel with best hyperparamter that fit your data')

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
            pycaret_report = st.checkbox('Pycaret Report')
            if pycaret_report:
                if problem == 'Regression':
                    RM = reg(df, target)
                    Rmodels= st.radio('Regression Models', ('None','LinearRegression','RidgeRegression','LassoRegression','ElasticNet', 'SupportVectorRegressor', 'DecisionTreeRegressor','RandomForestRegressor','GradientBoostingRegressor','xgboost','KNeighborsRegressor'))
    
                    if Rmodels == 'LinearRegression':
                        metrics ,preds = RM.Linear_Regression()
                        st.subheader('Linear Regression Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Rmodels == 'RidgeRegression':
                        metrics,preds  = RM.Ridge_Regression()
                        st.subheader('Ridge Regression Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')

                        
                    if Rmodels == 'LassoRegression':
                        metrics,preds  = RM.Lasso_Regression()
                        st.subheader('Lasso Regression Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')

                    if Rmodels == 'ElasticNet':
                        metrics,preds  = RM.Elastic_Net()
                        st.subheader('Linear Regression Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')

                    if Rmodels == 'SupportVectorRegressor':
                        metrics,preds  = RM.SVR()
                        st.subheader('Support VectorRegressor Regression Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
 
                    if Rmodels == 'DecisionTreeRegressor':
                        metrics,preds  = RM.DTR()
                        st.subheader('DecisionTreeRegressor Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Rmodels == 'RandomForestRegressor':
                        metrics,preds  = RM.RF()
                        st.subheader('RandomForestRegressor Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
            
                    if Rmodels == 'GradientBoostingRegressor':
                        metrics,preds  = RM.GBR()
                        st.subheader('GradientBoostingRegressorr Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                        
                    if Rmodels == 'xgboost':
                        metrics,preds =  RM.XGBOOST()
                        st.subheader('xgboost Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Rmodels == 'KNeighborsRegressor':
                        metrics,preds =  RM.KNN()
                        st.subheader('KNeighborsRegressor Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                    
                    show = st.checkbox('show Metrics table and Best Model..')
                    if show:
                        with st.spinner("Comparing Models..."):
                            metrics,preds = RM.model_metrics()
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')

                elif problem == 'Classification':
                    df = eda.label_encoder(target)
                    clf = classification(df, target)
                    Cmodels= st.radio('Classification Models', ('None','Logistic_Regressioin','KNearestNeighborsClassifier','NaiveBayes','DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtremeGradientBoostingClassifier','AdaBoostClassifier','LinearDiscriminantAnalysis','NeuralNetworkClassifier'
,'SupportVectorClassifier'))
                    if Cmodels == 'Logistic_Regressioin':
                        metrics,preds = clf.Logistic_Regression()
                        st.subheader('Logistic_Regressioin Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Cmodels == 'KNearestNeighborsClassifier':
                        metrics,preds = clf.KNN()
                        st.subheader('KNearestNeighborsClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Cmodels == 'NaiveBayes':
                        metrics,preds = clf.NB()
                        st.subheader('NaiveBayes Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Cmodels == 'DecisionTreeClassifier':
                        metrics,preds = clf.DT()
                        st.subheader('DecisionTreeClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                    
                    if Cmodels == 'RandomForestClassifier':
                        metrics,preds = clf.RF()
                        st.subheader('RandomForestClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Cmodels == 'GradientBoostingClassifier':
                        metrics ,preds= clf.GBC()
                        st.subheader('GradientBoostingClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                    if Cmodels == 'ExtremeGradientBoostingClassifier':
                        metrics,preds = clf.XGBOOST()
                        st.subheader('ExtremeGradientBoostingClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                    
                    if Cmodels == 'AdaBoostClassifier':
                        metrics,preds = clf.ADA()
                        st.subheader('AdaBoostClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                     
                    if Cmodels == 'LinearDiscriminantAnalysis':
                        metrics,preds = clf.LDA()
                        st.subheader('LinearDiscriminantAnalysis Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                    
                    if Cmodels == 'NeuralNetworkClassifier':
                        metrics,preds = clf.NN()
                        st.subheader('NeuralNetworkClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                    
                    if Cmodels == 'SupportVectorClassifier':
                        metrics,preds = clf.SVM()
                        st.subheader('SupportVectorClassifier Metrics')
                        st.write(metrics)
                        st.subheader('Evaluation of Test Data')
                        st.write(f'Accuracy Score in test data: {round(preds,2)}')
                        
                show = st.checkbox('show Metrics table and Best Model..')
                if show:
                    with st.spinner("Comparing Models..."):
                        metrics,preds = clf.model_metrics()
                    st.write(metrics)
                    st.subheader('Evaluation of Test Data')
                    st.write(f'Accuracy Score in test data: {round(preds,2)}')

                    
                    
                    
                    
                    
                    
                        
            
                    
                    
                            

if __name__ == '__main__':
    main()




