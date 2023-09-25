from pycaret.regression import *
from pycaret.classification import*

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd


class reg:
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.X_train, self.X_test, self.y_train,self.y_test =  train_test_split(self.data.drop(self.target,axis = 1),self.data[self.target],test_size=0.3, random_state=42, shuffle= True)
        self.reg_setup = setup(data=self.data, target=self.target, session_id=42,normalize = True)
    def Linear_Regression(self):
        lr = create_model('lr')
        metrics = pull()
        tuned = tune_model(lr)
        
        plot_model(tuned, plot = 'cooks',display_format='streamlit')
        plot_model(tuned, plot = 'residuals',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'rfe',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])

        return metrics, r2
        
    
    def Ridge_Regression(self):
        ridge = create_model('ridge')
        metrics = pull()
        tuned = tune_model(ridge)
        
        plot_model(tuned, plot = 'cooks',display_format='streamlit')
        plot_model(tuned, plot = 'residuals',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'rfe',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def Lasso_Regression(self):
        lasso = create_model('lasso')
        metrics = pull()
        tuned = tune_model(lasso)
        
        plot_model(tuned, plot = 'cooks',display_format='streamlit')
        plot_model(tuned, plot = 'residuals',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'rfe',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def Elastic_Net(self):
        en = create_model('en')
        metrics = pull()
        tuned = tune_model(en)
        
        plot_model(tuned, plot = 'cooks',display_format='streamlit')
        plot_model(tuned, plot = 'residuals',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'rfe',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def SVR(self):
        svr = create_model('svm')
        metrics = pull()
        tuned = tune_model(svr)
        
        plot_model(tuned, plot = 'residuals',display_format='streamlit')
        # plot_model(tuned, plot = 'prediction',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(svr, plot = 'feature',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def DTR(self):
        dt = create_model('dt')
        metrics = pull()
        
        plot_model(dt, plot = 'residuals',display_format='streamlit')
        plot_model(dt, plot = 'feature',display_format='streamlit')
        plot_model(dt, plot = 'error',display_format='streamlit')
        plot_model(dt, plot = 'learning',display_format='streamlit')
        plot_model(dt, plot = 'vc',display_format='streamlit')
        predictions = predict_model(df, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def RF(self):
        rf = create_model('rf')
        metrics = pull()
        
        plot_model(rf, plot = 'residuals',display_format='streamlit')
        plot_model(rf, plot = 'feature',display_format='streamlit')
        plot_model(rf, plot = 'error',display_format='streamlit')
        plot_model(rf, plot = 'learning',display_format='streamlit')
        plot_model(rf, plot = 'vc',display_format='streamlit')
        predictions = predict_model(rf, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
     
    def GBR(self):
        gbr = create_model('gbr')
        metrics = pull()        
        plot_model(gbr, plot = 'residuals',display_format='streamlit')
        plot_model(gbr, plot = 'feature',display_format='streamlit')
        plot_model(gbr, plot = 'error',display_format='streamlit')
        plot_model(gbr, plot = 'learning',display_format='streamlit')
        plot_model(gbr, plot = 'vc',display_format='streamlit')
        predictions = predict_model(gbr, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def XGBOOST(self):
        xgboost = create_model('xgboost')
        metrics = pull()
        # interprete = interpret_model(xgboost)
        
        plot_model(xgboost, plot = 'residuals',display_format='streamlit')
        plot_model(xgboost, plot = 'feature',display_format='streamlit')
        plot_model(xgboost, plot = 'error',display_format='streamlit')
        plot_model(xgboost, plot = 'learning',display_format='streamlit')
        plot_model(xgboost, plot = 'vc',display_format='streamlit')
        predictions = predict_model(xgboost, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def KNN(self):
        knn = create_model('knn')
        metrics = pull()
        # interprete = interpret_model(xgboost)
        
        plot_model(knn, plot = 'residuals',display_format='streamlit')
        # plot_model(knn, plot = 'feature',display_format='streamlit')
        plot_model(knn, plot = 'error',display_format='streamlit')
        plot_model(knn, plot = 'learning',display_format='streamlit')
        # plot_model(knn, plot = 'vc',display_format='streamlit')
        predictions = predict_model(knn, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
        return metrics,r2
    
    def model_metrics(self):
        best = compare_models()
        metrics_table = pull()
        plot_model(best, plot = 'residuals',display_format='streamlit')
        # plot_model(knn, plot = 'feature',display_format='streamlit')
        plot_model(best, plot = 'error',display_format='streamlit')
        plot_model(best, plot = 'learning',display_format='streamlit')
        # plot_model(knn, plot = 'vc',display_format='streamlit')
        predictions = predict_model(best, data=self.X_test)
        r2 = r2_score(self.y_test, predictions['prediction_label'])
      
        return metrics_table,r2
    
class classification:
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.X_train, self.X_test, self.y_train,self.y_test =  train_test_split(self.data.drop(self.target,axis = 1) ,self.data[self.target],test_size=0.3, random_state=42, shuffle= True)
        self.class_setup = setup(data=self.data, target=self.target, session_id=123,normalize = True)


    def Logistic_Regression(self):
        lr = create_model('lr')
        metrics = pull()
        tuned = tune_model(lr)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(lr, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        

        
    def KNN(self):
        knn = create_model('knn')
        metrics = pull()
        tuned = tune_model(knn)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
   
        predictions = predict_model(tuned, data=self.X_test)
        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc

    def NB(self):
        nb = create_model('nb')
        metrics = pull()
        tuned = tune_model(nb)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(best, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        return metrics,acc

    def DT(self):
        dt = create_model('dt')
        metrics = pull()
        tuned = tune_model(dt)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        
    def RF(self):
        rf = create_model('rf')
        metrics = pull()
        tuned = tune_model(rf)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature',display_format='streamlit')
   
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        return metrics,acc
        
    def GBC(self):
        gbc = create_model('gbc')
        metrics = pull()
        tuned = tune_model(gbc)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'feature_all',display_format='streamlit')
   
        predictions = predict_model(tuned, data=self.X_test)
        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        
    def XGBOOST(self):
        xgboost = create_model('xgboost')
        metrics = pull()
        tuned = tune_model(xgboost)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        
        predictions = predict_model(tuned, data=self.X_test)
        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        
    def ADA(self):
        ada = create_model('ada')
        metrics = pull()
        tuned = tune_model(ada)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        
    def LDA (self):
        lda = create_model('lda')
        metrics = pull()
        tuned = tune_model(lda)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)
        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc

    def NN(self):
        mlp = create_model('mlp')
        metrics = pull()
        tuned = tune_model(mlp)
        
        plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
        
    def SVM(self):
        svm = create_model('svm')
        metrics = pull()
        tuned = tune_model(svm)
        
        # plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(tuned, plot = 'pr',display_format='streamlit')
        plot_model(tuned, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(tuned, plot = 'error',display_format='streamlit')
        plot_model(tuned, plot = 'class_report',display_format='streamlit')
        plot_model(tuned, plot = 'boundary',display_format='streamlit')
        plot_model(tuned, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        predictions = predict_model(tuned, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        
        return metrics,acc
    
    def model_metrics(self):
        best = compare_models()
        metrics_table = pull()
        # plot_model(tuned, plot = 'auc',display_format='streamlit')
        plot_model(best, plot = 'pr',display_format='streamlit')
        plot_model(best, plot = 'confusion_matrix',display_format='streamlit')
        plot_model(best, plot = 'error',display_format='streamlit')
        plot_model(best, plot = 'class_report',display_format='streamlit')
        plot_model(best, plot = 'boundary',display_format='streamlit')
        plot_model(best, plot = 'learning',display_format='streamlit')
        # plot_model(tuned, plot = 'feature_all',display_format='streamlit')
        
        predictions = predict_model(best, data=self.X_test)

        acc = accuracy_score(self.y_test, predictions['prediction_label'])
        return metrics_table, acc
        
    
        

        
        