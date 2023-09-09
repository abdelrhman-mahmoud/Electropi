import numpy as np 
import pandas as pd 
 
from sklearn.metrics import accuracy_score, r2_score,confusion_matrix, mean_squared_error
from sklearn.metrics import recall_score,precision_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class Auto_ML:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
       
    
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
        best_model = grid_search.best_estimator_
        
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)
                }
        return scores
    
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



        best_model = grid_search.best_estimator_
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)
                }
        return scores
    
    def support_vector_regressor(self):
        param_grid = {'svr__C': [0.1, 1, 10],
        'svr__kernel': ['linear', 'rbf'],
        'svr__gamma': ['scale', 'auto']
        }


        pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('svr', SVR())
            ])
    
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train) 

        best_model = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)
                }
        return scores
    
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
      
        best_model = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
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
        grid_search.fit(self.X_train,self.y_train) 
        
        best_model = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores

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

        grid_search = GridSearchCV(pipeline,param_grid,cv=5)
        grid_search.fit(self.X_train,self.y_train)  
        best_model = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    
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

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5  
        )

        grid_search.fit(self.X_train, self.y_train)  
        best_model = grid_search.best_estimator_

        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    
    def log_regression(self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    

    def KNNC (self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    def DTC(self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    def RFC (self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    
    def svc(self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores
    
    def GBC(self):
        pipeline = Pipeline([
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
        scores = {
            'Best Hyperparameters':grid_search.best_params_,
            'Validation_scor':grid_search.best_score_,
            'Test_score':best_model.score(self.X_test, self.y_test)}
        return scores

