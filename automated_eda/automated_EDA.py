import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px 
import streamlit as st 
# import os 

class EDA:
    def __init__(self, df) :
        self.df = df 
        self.num_vars = self.df.select_dtypes(include=np.number).columns
        self.cat_vars = self.df.select_dtypes(include='object').columns
    
    def box_plot(self, var, col_x = None, hue=None):
        return px.box(self.df, x=col_x, y = var, color=hue)
    
    def histogram_(self, var):
        return  px.histogram(self.df, x=var)
    
    def scatter_plot(self, col_x,col_y):
        return px.scatter(self.df, x=col_x, y=col_y)

    def bar_plot(self, var):
        return self.df[var].value_counts().plot(kind = 'bar')


    def DistPlot(self, var):
        return sns.distplot(self.df[var], color='c', rug=True)
    

    def correlation (self,df):
        corr = df.corr() 
        return sns.heatmap(corr)

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
     return df.describe().transpose()

def wrangle(df):
    # Drop columns that are more than 25% null values.
    thresh = df.shape[0]//4
    df.dropna(inplace = True,thresh = thresh, axis = 1)
    
    # Remove outliers by trimming the bottom and top 10% of in each column
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            low, high = df[col].quantile([0.05,0.95])
            df = df[df[col].between(low, high)]

    # Drop columns containing low- or high-cardinality categorical values.
    df_cat = df.select_dtypes(include = 'object')
    LH_cardinality = [col for col in df_cat.columns if(df_cat[col].nunique() < 2 or df_cat[col].nunique() >20)]
    df_cat.drop(LH_cardinality,axis =1,inplace = True)

    # fill null values for categorical columns 
    for col in df_cat.columns:
        most_frequent_value = df_cat[col].mode()[0]
        df_cat[col].fillna(most_frequent_value, inplace=True)
            
    # fill null values for numerical columns 
    df_nums = df.select_dtypes(include = np.number)
    df_nums.fillna(df_nums.mean(),inplace = True)

    # scaling numerical feature
    df_nums.apply(lambda x: (x - np.mean(x)) / np.std(x))

    # encoding categorical features
    if df_cat.shape[1]>=1:
        df_cat = pd.get_dummies(df_cat,drop_first=True, dtype = float)

    # merge df_nums with df_cat
    df = pd.concat([df_nums, df_cat], axis = 1)
    return df

def plot_univariate(obj_plot, var, radio_plot_uni):
    
    if radio_plot_uni == 'Histogram' :
        st.subheader('Histogram')
        st.plotly_chart(obj_plot.histogram_(var))
    
    if radio_plot_uni == 'Distribution Plot':
        st.subheader('Distribution Plot')
        obj_plot.DistPlot(var)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()  

    if radio_plot_uni == 'BoxPlot' :
        st.subheader('Boxplot')
        st.plotly_chart(obj_plot.box_plot(var))

    if radio_plot_uni == 'bar plot':
        st.subheader('bar plot')
        obj_plot.bar_plot(var)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

def plot_multivariate(obj_plot, var2,var3):

    if var2 in obj_plot.num_vars and var3 in obj_plot.num_vars :
        st.subheader('scatter plot')
        st.plotly_chart(obj_plot.scatter_plot(var2,var3))

        


def main ():
    st.title('Exploratory Data Analysis')
    file  = st.file_uploader('Upload your file ', type = ['csv','xlsx'])
    
    if file is not None:
        df = get_data(file)

        numeric_features = df.select_dtypes(include=np.number).columns
        categorical_features = df.select_dtypes(include='object').columns
        
        choose = st.selectbox('show information about your data ',[None,'Original Data','preprocessed Data'])
        def info(df):
            st.header("about yor data")
            st.write('Number of observations', df.shape[0]) 
            st.write('Number of variables', df.shape[1])
            st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))

        # info(df)


        if choose == 'Original Data':
            info(df)
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

        elif choose == 'preprocessed Data':
            df = wrangle(df)
            info(df)
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

        st.header('Univariate Statistics')
        st.markdown("Provides summary statistics of only one variable.")
        var = st.selectbox('select one variable to analyze',  df.columns.insert(0,None))
        plot = EDA(df)
        if var in numeric_features:
            if var != None:
                chart_univariate = st.radio('Chart', ('None','Histogram', 'BoxPlot', 'Distribution Plot'))
                plot_univariate(plot, var, chart_univariate)
        if var in categorical_features:
            if var != None:
                chart_univariate = st.radio('Chart', ('None','bar plot','pie'))
                plot_univariate(plot, var, chart_univariate)

        st.header('multivariate Statistics')
        st.markdown("Provides summary statistics of tow variable.")
        var1 = st.radio('chart',('None','scatter plot', 'corrolation')) 
        if var1 == 'scatter plot':
            var2 = st.selectbox('select 1st numerical feature',df.columns.insert(0,None))
            var3 = st.selectbox('selcet 2nd numerical feature',df.columns.insert(0,None))
            plot_multivariate(plot, var2, var3)
        if var1 == 'corrolation':
            df_num = df.select_dtypes(include = np.number)
            fig = plot.correlation(df_num)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

if __name__ == '__main__':
    main()