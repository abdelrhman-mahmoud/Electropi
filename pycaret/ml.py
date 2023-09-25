
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score,confusion_matrix, mean_squared_error,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from imblearn.over_sampling import SMOTE
from EDA import plotting
from yellowbrick.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.grid(False)

fig = plotting()

class Auto_ML:
    def __init__(self,X,y,X_train,X_test,y_train,y_test,cols):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cols = cols
        self.X = X
        self.y = y
       
    
    def Linear_regression(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression()),
#             ('encoder', OneHotEncoder(drop= 'first'))
        ])

        param_grid = {
            'regressor__fit_intercept': [True, False]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose()         

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Linear Regression", self.X, self.y)
        plt.close(f)
        pipeline.fit(self.X_train, self.y_train)
        perm_importance = permutation_importance(pipeline, self.X_test, self.y_test, n_repeats=30, random_state=0)
        feature_importances = perm_importance.importances_mean
        f2 = fig.feature_importance_plot(feature_importances, self.cols)

        return scores,df,f,f2


    def ridge(self):
        pipeline = Pipeline(
            [('scaler', StandardScaler()),
            ('ridge', Ridge())
            ])

        param_grid = {
            'ridge__alpha': [0.1, 1.0, 10.0],
            'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Ridge ", self.X, self.y)
        pipeline.fit(self.X_train, self.y_train)
        perm_importance = permutation_importance(pipeline, self.X_test, self.y_test, n_repeats=30, random_state=0)
        feature_importances = perm_importance.importances_mean
        f2 = fig.feature_importance_plot(feature_importances, self.cols)

        return scores,df,f,f2

        
    
    def support_vector_regressor(self):
        param_grid = {'svr__C': [0.1, 1, 10],
        'svr__kernel': ['linear', 'rbf','poly'],
        'svr__gamma': ['scale', 'auto']
        }


        pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('svr', SVR())
            ])

        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Sunpport Vector Regressor", self.X, self.y)

        return scores,df,f

    
    def DTR(self):
        param_grid = {
            'dt__criterion': ['mse', 'friedman_mse', 'mae'],
            'dt__max_depth': [None, 5, 10],
            'dt__min_samples_split': [2, 5, 10]
        }


        pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('dt', DecisionTreeRegressor())
        ])

    
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Decision Tree Regressor", self.X, self.y)
        plt.close(f)
        pipeline.fit(self.X_train, self.y_train)
        feature_importances = pipeline.named_steps['dt'].feature_importances_
        f2 = fig.feature_importance_plot(feature_importances, self.cols)
        return scores,df,f,f2
    

    def KNNR(self):
        
        param_grid = {
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance'],
            'knn__p': [1, 2]
        }

        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('knn', KNeighborsRegressor())
        ])

     
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve K_nearst neigbour Regressor ", self.X, self.y)

        pipeline.fit(self.X_train,self.y_train)
        perm_importance = permutation_importance(pipeline, self.X_test, self.y_test, n_repeats=30, random_state=0)
        feature_importances = perm_importance.importances_mean
        f2 = fig.feature_importance_plot(feature_importances, self.cols)

        return scores,df,f,f2



    def RFR(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor())
        ])

        param_grid = {
            'regressor__n_estimators': range(2,20,2),
            'regressor__max_depth': [ 5, 10,20],
            'regressor__min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Random Forest Regressor", self.X, self.y)
        plt.close(f)
        pipeline.fit(self.X_train,self.y_train)
        feature_importances = pipeline.named_steps['regressor'].feature_importances_
        f2 = fig.feature_importance_plot(feature_importances, self.cols)

        return scores,df,f,f2

    
    def GBR(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor())
        ])

        
        param_grid = {
            'regressor__n_estimators': range(2,20,2),
            'regressor__learning_rate': [0.01, 0.1, 1.0],
            'regressor__max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_estimator = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_estimator.score(self.X_test, self.y_test)
        }

        y_pred = best_estimator.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        data = {
            'Value': [mae, mse,rmse, r2]
        }
        Metric =  ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error','R-squared']

        df = pd.DataFrame(data,index = Metric).transpose() 

        # Plot the learning curve
        f= fig.plot_learning_curve_R(best_estimator, "Learning Curve Gredient Boosting Regressor", self.X, self.y)

        plt.close(f)
        pipeline.fit(self.X_train, self.y_train)
        feature_importances = pipeline.named_steps['regressor'].feature_importances_
        f2 = fig.feature_importance_plot(feature_importances, self.cols)


        return scores,df,f,f2
    



    
    def log_regression(self):
        
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state= 42)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])

        # Define the parameter grid
        param_grid = {
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.1, 1, 10]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        f= fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)
        
        return scores, report,figure, f
    

    def KNNC (self):
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),  
            ('knn', KNeighborsClassifier())  
        ])
        param_grid = {
            'knn__n_neighbors': [3, 5, 7],
            'knn__weights': ['uniform', 'distance']  
        }
        # Create a GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        figure2= fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)

        return scores, report,figure,figure2

    def DTC(self):
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),  
            ('dt', DecisionTreeClassifier())  
        ])

        
        param_grid = {
            'dt__max_depth': [None, 5, 10],  
            'dt__min_samples_split': [2, 5, 10],  
            'dt__min_samples_leaf': [1, 2, 3]  
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        figure2= fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)


        return scores, report,figure,figure2

    def RFC (self):
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),  
            ('rf', RandomForestClassifier())  
        ])

        param_grid = {
            'rf__n_estimators': range(2,20,2), 
            'rf__max_depth': [None, 5, 10],  
            'rf__min_samples_split': [2, 5, 10],  
            'rf__min_samples_leaf': [1, 2, 3] 
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        figure2= fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)

        
        return scores, report,figure,figure2
    
    def svc(self):
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('svm', SVC())               
        ])
        param_grid = {
            
            'svm__C': [0.1, 1, 10],             
            'svm__kernel': ['linear', 'rbf'],  
            'svm__gamma': ['scale', 'auto']     
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        figure2 = fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)

        return scores, report,figure,figure2
    
    def GBC(self):
        pipeline = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),  
            ('gb', GradientBoostingClassifier())  
        ])

        param_grid = {
            'gb__n_estimators': range(2,20,2), 
            'gb__learning_rate': [0.01, 0.1, 0.2], 
            'gb__max_depth': [3, 4, 5],  
            'gb__min_samples_split': [2, 3]  
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        best_estimator = grid_search.best_estimator_

        y_pred = best_estimator.predict(self.X_test)
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        report =pd.DataFrame(classification_report(self.y_test, y_pred, target_names = np.unique(y_pred),output_dict=True))
        cm = confusion_matrix(self.y_test, y_pred)
        figure = fig.plot_confusion_matrix(cm,np.unique(y_pred))

        # fpr, tpr, thresholds = roc_curve(y_pred, self.y_test)
        # roc_auc = auc(fpr, tpr)
        # figure2 = fig.ORC_plot(fpr, tpr ,roc_auc)
        figure2= fig.plot_learning_curve_C(best_estimator, "Learning Curve Logisic Regression Classifier", self.X, self.y)


        return scores, report,figure,figure2
