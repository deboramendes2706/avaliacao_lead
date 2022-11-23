import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
import datetime 
import nltk
import pickle

def removing_white_spaces_comments(df):
    empty_reviews =[]
    for idx,label,review in df.itertuples():
        if type(review) == str:
            if review.isspace():
                empty_reviews.append(idx)
    print(empty_reviews)

    df.drop(empty_reviews, inplace=True)

    return df

def removing_ponctuation_and_numbers(df, column_init, colunm_final):
    df[colunm_final] = [re.sub(r"\d+", "" ,review) for review in df[column_init].tolist()]
    df[colunm_final] = [re.sub(r'[^\w\s]', '', review) for review in df[colunm_final].tolist()]
    df[colunm_final] = [re.sub(r'_+', ' ', review) for review in df[colunm_final].tolist()]
    df[colunm_final] = [re.sub(r"isnt", "is_not", review) for review in df[colunm_final].tolist()]

    return df

def removing_stop_words(df, init_column, final_column):
    stop_words = stopwords.words('english')
    review_no_stop_words = []
    for line in df[init_column]:
        words = word_tokenize(line)

        filtered = [word for word in words if not word.lower() in stop_words]

        review_no_stop_words.append(" ".join(filtered))
    df[final_column] = review_no_stop_words
    
    return df

def identifying_frequency(df,column):
    words = df[column].tolist()
    counts = Counter(" ".join(words).split()).most_common(30)
    words_frequency = pd.DataFrame(counts, columns = ["Words", "Frequency"])

    sns.set(rc={'figure.figsize':(25.0,5.0)})
    sns.barplot(x="Words", y="Frequency", data=words_frequency)


def lemmatizing_sentences(df, init_column, final_column):
    lem = nltk.WordNetLemmatizer()
    review_no_stop_words = []
    for line in df[init_column]:
        new_line = []
        words = word_tokenize(line)
        for word in words: 
            new_line.append(lem.lemmatize(word))
        review_no_stop_words.append(" ".join(new_line))
    df[final_column] = review_no_stop_words
    return df

def stemming_sentences(df, init_column, final_column):
    porter = nltk.PorterStemmer()
    review_no_stop_words = []
    for line in df[init_column]:
        new_line = []
        words = word_tokenize(line)
        for word in words: 
            new_line.append(porter.stem(word))
        review_no_stop_words.append(" ".join(new_line))
    df[final_column] = review_no_stop_words
    return df

def testing_models_random(dict_models, dict_vetorizadores, dict_param_vetorizadores, x_train, y_train, x_test, y_test):
    df = pd.DataFrame(columns = ["Modelo", "acc", "F1-score"])
    for nome_vetorizador,vetorizador in dict_vetorizadores.items():
        for nome_modelo, modelo in dict_models.items():
            print(f"Iniciou o modelo: {nome_modelo}_{nome_vetorizador}")
            print(datetime.datetime.now())
            if nome_modelo == "Bayes":
                modelo = {
                "clf" :  MultinomialNB(), #Utilizado quando temos texto
                        'parameters' : {}
                } 
            dict_results = {}
            pipeline = Pipeline([
                ('vetorizador',vetorizador),
                ('modelo',modelo["clf"])
            ])

            params = {}
            for name_parameter, parameter in modelo["parameters"].items():
                params[f"modelo__{name_parameter}"] = parameter

            params.update(dict_param_vetorizadores)

            rd = RandomizedSearchCV(pipeline,param_distributions=params, cv=5, n_jobs = -1, n_iter=5, verbose=2) 
            rd.fit(x_train,y_train) 

            y_pred = rd.predict(x_test)
            dict_results['Modelo']= f"{nome_modelo}_{nome_vetorizador}"
            dict_results['acc'] = accuracy_score(y_test,y_pred)
            dict_results['F1-score'] = f1_score(y_test,y_pred)

            df = pd.concat([df, pd.DataFrame([dict_results])], ignore_index = True)
            print(f"Finalizou o modelo: {nome_modelo}_{nome_vetorizador}")
            print(datetime.datetime.now())
    return df

def testing_models_grid(dict_models, dict_vetorizadores, dict_param_vetorizadores, df_train, df_test, columns):
    df_total = pd.DataFrame(columns = ["Modelo", "acc", "F1-score"])
    for column in columns:
        x_train = df_train[column].tolist()
        x_test = df_test[column].tolist()
        y_train = df_train["Toxic"].tolist()
        y_test = df_test["Toxic"].tolist()
        print(f"Coluna: {column}")
        df_for_column = pd.DataFrame(columns = ["Modelo", "acc", "F1-score"])
        for nome_vetorizador,vetorizador in dict_vetorizadores.items():
            for nome_modelo, modelo in dict_models.items():
                print(f"Iniciou o modelo: {nome_modelo}_{nome_vetorizador}")
                print(datetime.datetime.now())
                if nome_modelo == "Bayes":
                    modelo = {
                    "clf" :  MultinomialNB(), #Utilizado quando temos texto
                            'parameters' : {}
                    } 
                dict_results = {}
                pipeline = Pipeline([
                    ('vetorizador',vetorizador),
                    ('modelo',modelo["clf"])
                ])

                params = {}
                for name_parameter, parameter in modelo["parameters"].items():
                    params[f"modelo__{name_parameter}"] = parameter

                params.update(dict_param_vetorizadores)

                rd = GridSearchCV(pipeline,param_grid=params, cv=10, n_jobs = -1, verbose=2) 
                rd.fit(x_train,y_train) 

                with open(f'{nome_modelo}_{nome_vetorizador}_{column}_best_model.pickle', 'wb') as f:
                    pickle.dump(rd,f)

                y_pred = rd.predict(x_test)
                dict_results['Modelo']= f"{nome_modelo}_{nome_vetorizador}_{column}"
                dict_results['acc'] = accuracy_score(y_test,y_pred)
                dict_results['F1-score'] = f1_score(y_test,y_pred)

                df_for_column = pd.concat([df_for_column, pd.DataFrame([dict_results])], ignore_index = True)
                df_total = pd.concat([df_total, pd.DataFrame([dict_results])], ignore_index = True)
                print(f"Finalizou o modelo: {nome_modelo}_{nome_vetorizador}_{column}")
                print(datetime.datetime.now())
        print("--------------------------------------------")
        new_df = df_for_column.to_csv(index=False)
        with open(f"df_{column}.csv", 'w') as file:
            file.write(new_df)
        display(df_for_column)
    new_df = df_total.to_csv(index=False)
    with open(f"df_total_results.csv", 'w') as file:
        file.write(new_df)
    display(df_total)
    return df_total