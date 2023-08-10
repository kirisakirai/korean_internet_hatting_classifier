class model_eval:
    def __init__(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # 모델 학습
    def model_train(self, cv=None):
        from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegressionCV, Ridge
        from sklearn.svm import SVC, SVR, NuSVC, NuSVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
        from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, \
            gaussian_process
        from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, \
            f1_score, roc_auc_score
        from sklearn.model_selection import cross_validate, GridSearchCV, ParameterGrid
        import pandas as pd
        print('X_train.shape :', self.X_train.shape)
        print('y_train.shape :', self.y_train.shape)
        print('X_test.shape :', self.X_test.shape)
        print('y_test.shape :', self.y_test.shape)
        self.model.fit(self.X_train, self.y_train)
        self.pred = self.model.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.pred)
        print('모델 이름:', self.model.__class__.__name__)
        # params_dict = self.model.get_params()  # 딕셔너리 형태의 파라미터 가져오기
        # params_df = pd.DataFrame(list(params_dict.items()), columns=['파라미터', '값'])
        # self.display_scores(params_df, mean=None)
        print('train_Accuracy_score(훈련정확도) :', accuracy_score(self.y_train, self.model.predict(self.X_train)))
        if isinstance(self.model, GridSearchCV):
            self.model.fit(self.X_train, self.y_train)
            print(self.model.best_score_)
            print(self.model.best_params_)
            return self.model
        else:
            if hasattr(self.model, 'predict_proba'):
                self.pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                self.model_test(cv)
            else:
                self.pred_proba = None
                self.model_test(cv)
        return self.model

    def print_scores(self, y_test, pred, pred_proba=None):
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
            roc_auc_score
        cm = confusion_matrix(y_test, pred)
        TN, FP, FN, TP = cm.ravel()
        print('오차행렬 \n', cm)
        print('Accuracy_score(정확도) :', accuracy_score(y_test, pred))
        print('Precision(정밀도) : ', precision_score(y_test, pred))
        print('Recall(재현율) :', recall_score(y_test, pred))
        print('TNR(0을 맞춘 비율) :', TN / (TN + FP))
        print('F1 score :', f1_score(y_test, pred))
        if pred_proba is not None:
            print('Roc Auc score :', roc_auc_score(y_test, pred_proba))

    def model_test(self, cv=None):
        from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, \
            f1_score, roc_auc_score
        from sklearn.model_selection import cross_validate

        if cv is not None:
            self.cross_validation(cv)
        else:
            self.pred = self.model.predict(self.X_test)
            if hasattr(self.model, 'predict_proba'):
                self.pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            else:
                self.pred_proba = None
            self.print_scores(self.y_test, self.pred, self.pred_proba)

    def cross_validation(self, cv):
        from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, \
            f1_score, roc_auc_score
        from sklearn.model_selection import cross_validate
        import pandas as pd
        def TNR(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average=cv),
            'recall': make_scorer(recall_score, average=cv),
            'f1': make_scorer(f1_score, average=cv),
            'TNR': make_scorer(TNR)
        }

        if hasattr(self.model, 'predict_proba'):
            scoring['roc_auc'] = make_scorer(roc_auc_score, needs_proba=True)

        scores = cross_validate(self.model, self.X_train, self.y_train, cv=5, scoring=scoring)
        scores_df = pd.DataFrame(scores).round(3)
        scores_df.rename(columns={
            'test_accuracy': 'Accuracy',
            'test_precision': 'Percision',
            'test_recall': 'Recall',
            'test_f1': 'F1',
            'test_TNR': 'TNR',
            'test_roc_auc': 'RocAuc'}, inplace=True)
        self.display_scores(scores_df, mean=True)

    def display_scores(self, df, mean=None):
        import pandas as pd
        def is_ipython():
            try:
                __IPYTHON__
                return True
            except NameError:
                return False

        if is_ipython():
            from IPython.display import display
            display(df)
            if mean is not None:
                print()
                display(df.mean())
        else:
            print(df)
            if mean is not None:
                print()
                print(df.mean())

    def model_visual(self, cm_heatmap='b'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib as mpl

        if cm_heatmap is not None:
            if cm_heatmap == 'b':
                new_confusion = [[self.cm[1, 1], self.cm[1, 0]], [self.cm[0, 1], self.cm[0, 0]]]

                plt.figure(figsize=(5, 5))
                sns.heatmap(new_confusion, annot=True, fmt=".0f", linewidths=.5, square=True,
                            cmap=sns.diverging_palette(220, 10, as_cmap=True),
                            xticklabels=['1', '0'], yticklabels=['1', '0']);

                plt.ylabel('Acutral label')
                plt.xlabel('Predicted label')

                title = 'Confusion Matrix'
                plt.title(title, size=15);
                plt.show()

            elif cm_heatmap == 'c':
                new_confusion = [[self.cm[1, 1], self.cm[0, 0]], [self.cm[1, 0], self.cm[0, 1]]]

                plt.figure(figsize=(5, 5))
                sns.heatmap(new_confusion, annot=True, fmt=".0f", linewidths=.5, square=True,
                            cmap=sns.diverging_palette(220, 10, as_cmap=True),
                            xticklabels=['1', '0'], yticklabels=['Correct', 'Incorrect']);

                plt.ylabel('Correct')
                plt.xlabel('Predicted')

                title = 'Confusion Matrix'
                plt.title(title, size=15);
                plt.show()

    def model_save(self):
        model = self.model
        save_type = input('save type : ')
        if save_type == 'pickle':
            import pickle
            save_path = input('save path+name.pickle: ')
            protocol = input('protocol (default pickle.HIGHEST_PROTOCOL, press enter to use default) : ')
            if protocol == '':
                print('디폴트')
                protocol = pickle.HIGHEST_PROTOCOL
            else:
                print('입력')
                protocol = int(protocol)
            with open(save_path, 'wb') as handle:
                pickle.dump(model, handle, protocol=protocol)
        elif save_type == 'joblib':
            import joblib
            save_path = input('save path+name.joblib :')
            joblib.dump(model, save_path)
        return print(f'{model}을 {save_path}에 저장했습니다.')

    def model_load(self):
        type = input('load type :')
        if type == 'pickle':
            import pickle
            load_path = input('load path+name.pickle :')
            with open(load_path, 'rb') as handle:
                self.model = pickle.load(handle)
        elif type == 'joblib':
            import joblib
            load_path = input('load path+name.joblib :')
            self.model = joblib.load(load_path)
        return self.model, print(f'{self.model}을 {load_path}에서 불러왔습니다.')

    def transformer_save(self, transformer):
        save_type = input('save type :')
        if save_type == 'pickle':
            import pickle
            save_path = input('save path+name.pickle:')
            protocol = input('protocol (default pickle.HIGHEST_PROTOCOL, press enter to use default) : ')
            if protocol == '':
                protocol = pickle.HIGHEST_PROTOCOL
            else:
                protocol = int(protocol)
            with open(save_path, 'wb') as handle:
                pickle.dump(transformer, handle, protocol=protocol)
        elif save_type == 'joblib':
            import joblib
            save_path = input('save path+name.joblib :')
            joblib.dump(transformer, save_path)
        return print(f'{transformer}을 {save_path}에 저장했습니다.')

    def transformer_load(self, transformer):
        type = input('load type :')
        if type == 'pickle':
            import pickle
            load_path = input('load path+name.pickle :')
            with open(load_path, 'rb') as handle:
                trans = pickle.load(handle)
        elif type == 'joblib':
            import joblib
            load_path = input('load path+name.joblib :')
            transformer = joblib.load(load_path)
        return transformer, print(f'{transformer}을 {load_path}에서 불러왔습니다.')

    # 기본


import pandas as pd


# 모델
import sklearn.svm as SVC


import os

path = os.getcwd()

train = pd.read_csv(path + "/스마게 언스마일 train.CSV", encoding='cp949')
test = pd.read_csv(path + "/스마게 언스마일 test.CSV", encoding='cp949')

train['target'] = train['clean'].replace((1, 0), (0, 1))
test['target'] = test['clean'].replace((1, 0), (0, 1))

with open(path + "/깨끗한대화35000개.txt", 'r', encoding='cp949') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines]

add_data = pd.DataFrame(lines)
add_data['target'] = 0
add_data.rename(columns={0: '문장'}, inplace=True)

X_train = train['문장']
y_train = train['target']
X_test = test['문장']
y_test = test['target']

X_train = pd.concat([X_train, add_data['문장'].head(7656)])
y_train = pd.concat([y_train, add_data['target'].head(7656)])

X_test = pd.concat([X_test, add_data['문장'].tail(1881)])
y_test = pd.concat([y_test, add_data['target'].tail(1881)])

import rhinoMorph

rn = rhinoMorph.startRhino()
train_morphed_data_each = []
test_morphed_data_each = []
data_all = [X_train, X_test]
for idx, val in enumerate(data_all):
    for sentence in val:
        if idx == 0:
            train_morphed_data_each.append(
                rhinoMorph.onlyMorph_list(rn, sentence, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'],
                                          eomi=True))
            train_X_join = [" ".join(sentence) for sentence in train_morphed_data_each]
        elif idx == 1:
            test_morphed_data_each.append(
                rhinoMorph.onlyMorph_list(rn, sentence, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'],
                                          eomi=True))
            test_X_join = [" ".join(sentence) for sentence in test_morphed_data_each]

print(train_morphed_data_each[:2])
print(test_morphed_data_each[:2])
print(train_X_join[:2])
print(test_X_join[:2])

from sklearn.feature_extraction.text import CountVectorizer

vect_morp = CountVectorizer(max_features=10000).fit(train_X_join)
X_train_final = vect_morp.transform(train_X_join)
X_test_final = vect_morp.transform(test_X_join)
print('X_train:\n', repr(X_train_final))
print('X_test:\n', repr(X_test_final))

from sklearn.svm import SVC

eval = model_eval(SVC(C=7350, gamma='auto', kernel='rbf', probability=True), X_train_final, y_train, X_test_final,
                  y_test)
eval.model_train()

eval.model_save()
eval.transformer_save(vect_morp)