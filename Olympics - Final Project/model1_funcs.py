# We will include most of the functionality used for the first model in this .py file in order to keep the main notebook
# as clean as possible

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate
)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def show_missing_evolution(df):
    """
    Function to show evolution of feature variables over time

    :param df: Dataframe
    :return: Plot
    """

    df_height_missing = df.Height.isnull().groupby(df['Year']).sum().astype(int).reset_index(name='Height')
    df_weight_missing = df.Weight.isnull().groupby(df['Year']).sum().astype(int).reset_index(name='Weight')
    df_age_missing = df.Age.isnull().groupby(df['Year']).sum().astype(int).reset_index(name='Age')
    df_year_count = df.groupby(df['Year']).Name.count().astype(int).reset_index(name='Count')

    df_missing_year = df_age_missing.merge(df_weight_missing, on='Year')
    df_missing_year = df_missing_year.merge(df_height_missing, on='Year')
    df_missing_year = df_missing_year.merge(df_year_count, on='Year')

    df_missing_year['Age'] = df_missing_year.Age / df_missing_year.Count
    df_missing_year['Weight'] = df_missing_year.Weight / df_missing_year.Count
    df_missing_year['Height'] = df_missing_year.Height / df_missing_year.Count

    df_missing_year.plot(x='Year', y=['Age', 'Height', 'Weight'], figsize=(12, 8))
    plt.ylabel('Percentage of Missing Values')
    plt.show()


def data_preprocessing(df):
    """
    Function to apply all pre-processing steps required

    :param df: Dataframe
    :return: Clean dataframe
    """

    # Remove observations before 1960
    df_clean = df[df.Year >= 1960]

    # Split Athletics sport into sub-sports based on events.
    df_clean[df_clean.Sport == 'Athletics'].Event.unique().tolist()
    df_clean.loc[(df_clean["Sport"] == "Athletics") & (
        df_clean["Event"].str.contains("jump|vault|60 |100 |200 |400 |athlon|all-round", case=False)),
                 "Sport"] = "Athletics Sprints"
    df_clean.loc[(df_clean["Sport"] == "Athletics") & (
        df_clean["Event"].str.contains("put|throw", case=False)),
                 "Sport"] = "Athletics Throws"
    df_clean.loc[(df_clean["Sport"] == "Athletics") & ~(
        df_clean["Event"].str.contains("jump|vault|60 |100 |200 |400 |athlon|all-round", case=False)) & ~(
        df_clean["Event"].str.contains("put|throw", case=False)),
                 "Sport"] = "Athletics Endurance"

    # Replace missing values with averages at sport, sex and year level
    df_clean['Age'] = df_clean.groupby(['Year', 'Sport', 'Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
    df_clean['Height'] = df_clean.groupby(['Year', 'Sport', 'Sex'])['Height'].transform(lambda x: x.fillna(x.mean()))
    df_clean['Weight'] = df_clean.groupby(['Year', 'Sport', 'Sex'])['Weight'].transform(lambda x: x.fillna(x.mean()))
    df_clean = df_clean[~df_clean.Height.isnull()]
    df_clean = df_clean[~df_clean.Weight.isnull()]

    # Focus on summer sports
    df_clean = df_clean[df_clean.Season == 'Summer']

    # Only keep sports that have been played during the 2016 Olympics
    sports_2016 = df_clean[df_clean.Year == 2016].Sport.unique().tolist()
    df_clean = df_clean[df_clean.Sport.isin(sports_2016)]

    # Construct Target Variable
    df_clean['Target_Variable'] = np.where(df_clean['Medal'].isnull(), 0, 1)

    # Merge teams into single athletes
    #       1. Identify team sports (this will be done approximately, i.e. we will group by year and event type. Those
    #          events for which more than 4 medals have been given out will be assumed to be team sports). We will
    #          take the value 4 since sometimes two silver medals or gold could be given out

    team_sports = df_clean.groupby(['Year', 'Event']).Target_Variable.sum().reset_index(name='count')
    team_sport_list = team_sports[team_sports['count'] > 4].Event.unique().tolist()

    df_clean['sport_type'] = ['team' if event in team_sport_list else 'individual' for event in df_clean.Event.tolist()]

    #        2. Do the groupby such that team sports are grouped by year, NOC and event to get the average athlete
    #           of that year, event and team

    df_individual = df_clean[df_clean.sport_type == 'individual']
    df_team = df_clean[df_clean.sport_type == 'team']

    df_team_grouped = df_team.groupby(['Year', 'NOC', 'Event']).agg({'Age': 'mean',
                                                                     'Height': 'mean',
                                                                     'Weight': 'mean'})

    df_team_grouped.reset_index(level=['Year', 'NOC', 'Event'], inplace=True)

    df_team = df_team_grouped.merge(df_team.groupby(['Year', 'NOC', 'Event']).head(1).reset_index(drop=True),
                                    on=['Year', 'NOC', 'Event'], suffixes=(None, '_individual'))

    df_team.drop(columns=['Age_individual', 'Height_individual', 'Weight_individual'], inplace=True)

    df_clean = pd.concat((df_individual, df_team), ignore_index=True)

    # Compute the number of medals won by country, sport and sex over the whole olympic history (since 1960)
    df_clean['Medals_per_country'] = df_clean.groupby(['Sport', 'NOC', 'Sex'])['Target_Variable'].transform(
        lambda x: x.sum())
    df_clean = df_clean.sort_values(by=['Year'])

    return df_clean


def generate_mapping_table_country_sport_sex(df):
    """
    Function which returns a dictionary containing the number of medals won for each sport, country and sex (will be
    used to access this information quickly later on in the analysis)

    :param df: Dataframe
    :return: Dictionary
    """

    mapping_table = df.groupby(['Sport', 'NOC', 'Sex'])['Target_Variable'].sum().reset_index(name='medal_count')

    medals_country_dict = {}
    for sport in list(mapping_table.Sport.unique()):
        temp_df = mapping_table[mapping_table.Sport == sport]
        country_sex_dict = {}
        for country in list(temp_df.NOC.unique()):
            country_sex_dict[country] = dict(zip(temp_df[temp_df.NOC == country].Sex,
                                                 temp_df[temp_df.NOC == country].medal_count))
        medals_country_dict[sport] = country_sex_dict

    return medals_country_dict


def standardize_features(df, features=('Age', 'Height', 'Weight', 'Medals_per_country'), new_features=None):
    """
    Function to standardize features

    :param df: Dataframe to standardize
    :param features: Features to standardize
    :param new_features: If we have a set of features we want to apply our model on but are still not standardized (they
    need to be integers and in the same order as the features list)
    :return: Dataframe with standardized features or a list with some input variables also standarized
    """

    if not new_features:
        df_temp = df.copy()
        for feature in features:
            df_temp[feature] = (df_temp[feature] - np.mean(df_temp[feature])) / np.std(df_temp[feature])
        return df_temp
    else:
        i = 0
        standardized_features = []
        for feature in features:
            temp = (new_features[i] - np.mean(df[feature])) / np.std(df[feature])
            standardized_features.append(temp)
            i += 1
        return standardized_features


def generate_sample_split(df):
    """
    Function which will split a given input dataframe in test and train data sets.
    This function specifically will convert 2016 olympic games into the test sample and the rest as train sample.

    :param df: Dataframe
    :return: Test and Train data samples
    """
    # Training sample will be all the olympics before 2016
    df_train = df[df.Year < 2016]

    # Test sample will be the last olympics
    df_test = df[df.Year == 2016]

    # Remove Golf, 2016 is the first time this was an olympic sport
    df_test = df_test[df_test.Sport != 'Golf']
    df_test = df_test[df_test.Sport != 'Rugby Sevens']

    return df_train, df_test


def generate_sample_split_advanced(df):
    """
    This function will  split the data in test and train, and differenciate between sex

    :param df: Dataframe
    :return: Test and Train data samples
    """
    df_train_Female_X = df[(df.Sex == 'F') & (df.Year < 2016)][
        ['Year', 'Age', 'Height', 'Weight', 'Medals_per_country']]
    df_train_Male_X = df[(df.Sex == 'M') & (df.Year < 2016)][
        ['Year', 'Age', 'Height', 'Weight', 'Medals_per_country']]

    df_train_Female_Y = df[(df.Sex == 'F') & (df.Year < 2016)]['Sport_label']
    df_train_Male_Y = df[(df.Sex == 'M') & (df.Year < 2016)]['Sport_label']

    df_test_Female_X = df[(df.Sex == 'F') & (df.Year == 2016)][
        ['Year', 'Age', 'Height', 'Weight', 'Medals_per_country']]
    df_test_Male_X = df[(df.Sex == 'M') & (df.Year == 2016)][
        ['Year', 'Age', 'Height', 'Weight', 'Medals_per_country']]

    df_test_Female_Y = df[(df.Sex == 'F') & (df.Year == 2016)]['Sport_label']
    df_test_Male_Y = df[(df.Sex == 'M') & (df.Year == 2016)]['Sport_label']

    return df_train_Female_X, df_train_Male_X, df_train_Female_Y, df_train_Male_Y, df_test_Female_X, df_test_Male_X, \
           df_test_Female_Y, df_test_Male_Y


def train_model_target_win_medal(df, sex='F'):
    """
    Train models for a given sex (outputs should be very different based on sex due to different physical features).
    Input df will have a set of features over which the model will do its
    predicitions + ['Sex', 'Target_Variable', 'Sport'].
    The last three columns are used to help train the model but will not be used to train the model itself

    E.g. df = df_train[['Age', 'Height', 'Weight', 'Medals_per_country', 'Sex', 'Target_Variable', 'Sport'].

    :return: Dataframe with the theta values for a given bag of feature values (number of columns will be equal to the
    number of features + 1 (intercept) and the number of rows will be equal to the number of sports in df
    """

    features = df.columns.tolist()
    features = [feature for feature in features if feature not in ['Sex', 'Target_Variable', 'Sport']]

    theta_dict = {}
    df_train = df.copy()
    df_train = df_train[df_train.Sex == sex]
    for sport in df_train.Sport.unique():
        # Generate training data and target variable for the given sport
        X = df_train[df_train.Sport == sport][features]
        y = df_train[df_train.Sport == sport].Target_Variable

        # Initialize the Logistic Regression Model using sklearn
        logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000)

        # Train the model with the given data
        logistic_regression.fit(X, y)

        # Append intercepts and coefficients to the dictionary
        theta_dict[sport] = logistic_regression.intercept_.tolist() + logistic_regression.coef_[0].tolist()

    theta_df = pd.DataFrame.from_dict(theta_dict, orient='index')
    theta_df.columns = ['Intercept'] + features
    theta_df = theta_df.sort_index()

    return theta_df


def extend_dataset_features(dfs):
    """
    This function will take a dataframe (elements in list dfs) and extend the features accordingly

    :param dfs: List with dataframes over which we want to extend the features
    :return: Dict of extended dataframes
    """

    extended_dfs_dict = []

    for df in dfs:
        df_temp = df.copy()

        df_temp['Age^2'] = df_temp.Age ** 2
        df_temp['Height^2'] = df_temp.Height ** 2
        df_temp['Weight^2'] = df_temp.Weight ** 2
        df_temp['Medals_per_country^2'] = df_temp.Medals_per_country ** 2
        df_temp['Age*Height'] = df_temp.Age * df_temp.Height
        df_temp['Age*Weight'] = df_temp.Weight * df_temp.Age
        df_temp['Age*Medals_per_country'] = df_temp.Medals_per_country * df_temp.Age
        df_temp['Height*Weight'] = df_temp.Weight * df_temp.Height
        df_temp['Height*Medals_per_country'] = df_temp.Medals_per_country * df_temp.Height
        df_temp['Weight*Medals_per_country'] = df_temp.Medals_per_country * df_temp.Weight
        df_temp['Age^3'] = df_temp.Age ** 3
        df_temp['Height^3'] = df_temp.Height ** 3
        df_temp['Weight^3'] = df_temp.Weight ** 3
        df_temp['Medals_per_country^3'] = df_temp.Medals_per_country ** 3

        extended_dfs_dict.append(df_temp)

    return extended_dfs_dict


def optimal_feature_combinations(df_train, features_permutations_dict, model_function=train_model_target_win_medal,
                                 print_evolution=True):
    """
    Given a training data set and a dictionary with possible feature combinations, train the logictis regression model
    and store all the resulting theta values in a dictionary

    :param df_train: Data set over which the model will be trained
    :param features_permutations_dict: Set of potential features
    :param model_function: Function that will train the model
    :param print_evolution: If set to true the function will return some prints
    :return: Dictionary of dataframes
    """

    hyperparameter_thetas = {}

    for i, features in enumerate(list(features_permutations_dict.keys())):

        training_set_columns = features_permutations_dict[features] + ['Sex', 'Target_Variable', 'Sport']
        df = df_train.copy()
        df = df[training_set_columns]
        if print_evolution:
            print('Model ' + str(i + 1))
            print('Features inspected:', features_permutations_dict[features])
        sex_dict = {}
        for sex in ['F', 'M']:
            if print_evolution:
                print('Sex:', sex)
            try:
                sex_dict[sex] = model_function(df, sex)
            except:
                print('Model did not converge')
                sex_dict[sex] = None
        hyperparameter_thetas[features] = sex_dict
        if print_evolution:
            print('====================')

    return hyperparameter_thetas


def sigmoid(x):
    """
    Sigmoid function

    :param x: Vector with feature values
    :return: Sigmoid function values
    """
    return 1 / (1 + np.exp(-x))


def predict_logistic_regression(sport, new_features, theta_df, df_clean=None, standardize=True,
                                sigmoid_function=sigmoid):
    """

    Given a set of features and a given sport, compute the probability of being succesful in that sport (i.e.
    probability of winning a medal given a set of features and theta values)

    Warning: It is important to provide features in the correct order and they should already be standardized!!

    theta_df: Dataframe containing all the theta parameters for each feature
    df_clean: Only used if new_features are not standardized and want to be standardized
    new_features: need to be all integers and in the correct order with respect to theta_df
    """

    # Standardize features since model is computed based on standardized values
    if standardize:
        standardized_features = standardize_features(df_clean, new_features=new_features)
    else:
        standardized_features = new_features

    # Get model theta values
    theta_vector = theta_df.loc[sport].values
    X = [1] + standardized_features
    z = np.dot(X, theta_vector)

    # Compute probability
    h = sigmoid_function(z)

    return h


def model_performance(df_test_extended, hyperparameter_thetas,
                      predict_logistic_regression_model=predict_logistic_regression,
                      print_evolution=True):
    """
    Compute the performance of our different logistic regression models (stored in model_features) over the test data set
    (df_test_extended) computing accuracy, recall, precision and f1-score. The optimal model will finally be evaluated
    using the f1-score

    :param df_test_extended: Test data set with all features
    :param hyperparameter_thetas: Theta values for each model, sport and sex
    :param predict_logistic_regression_model: Function that applies the logistic regression
    :param print_evolution: If set to true the function will return some prints
    :return: Data frame will f1-score for female and male (at sport level)
    """

    validation_df_male = []
    validation_df_female = []

    model_features = list(hyperparameter_thetas.keys())

    for i, feature in enumerate(model_features):
        if print_evolution:
            print(feature)

        for sex in ['M', 'F']:
            if print_evolution:
                print('Sex: ', sex)
            df_test_temp = df_test_extended.copy()
            df_test_temp = df_test_temp[df_test_temp.Sex == sex]

            # Compute the probability of each athlete from the test data based on the features
            # hyperparameter_thetas[feature][sex]
            df_test_temp['predict_prob'] = df_test_temp.apply(
                lambda x: predict_logistic_regression_model(sport=x.Sport,
                                                            theta_df=hyperparameter_thetas[feature][sex],
                                                            new_features=x[
                                                                hyperparameter_thetas[feature][sex].columns.tolist()[
                                                                1:]].values.tolist(),
                                                            standardize=False),
                axis=1)

            # Given the probabilities choose then top3 athletes or those that have a probability of 1
            temp = df_test_temp.groupby(['Sport']).apply(lambda x: x.nlargest(3, ['predict_prob'])).reset_index(
                drop=True)
            temp['Medal_predicted'] = 1
            temp = temp[['Name', 'Event', 'Medal_predicted']]
            final = temp.merge(df_test_temp, on=['Name', 'Event'], how='outer')
            final['Medal_predicted'] = final.apply(lambda x: 1 if x.predict_prob == 1 else x.Medal_predicted, axis=1)
            final['Medal_predicted'].fillna(0, inplace=True)

            # Compute False/True positives/negatives
            final['True Positive'] = final.apply(
                lambda x: 1 if (x.Medal_predicted == 1) & (x.Target_Variable == 1) else 0, axis=1)
            final['True Negative'] = final.apply(
                lambda x: 1 if (x.Medal_predicted == 0) & (x.Target_Variable == 0) else 0, axis=1)
            final['False Positive'] = final.apply(
                lambda x: 1 if (x.Medal_predicted == 1) & (x.Target_Variable == 0) else 0, axis=1)
            final['False Negative'] = final.apply(
                lambda x: 1 if (x.Medal_predicted == 0) & (x.Target_Variable == 1) else 0, axis=1)

            accuracy = final.groupby('Sport').agg({'True Positive': 'sum', 'True Negative': 'sum',
                                                   'False Positive': 'sum', 'False Negative': 'sum'})

            # Compute recall, precision and f1-score. We will ultimately use the f1-score to choose between the ideal
            # features of our model
            accuracy['recall'] = accuracy['True Positive'] / (accuracy['True Positive'] + accuracy['False Negative'])
            accuracy['precision'] = accuracy['True Positive'] / (accuracy['True Positive'] + accuracy['False Positive'])
            accuracy['f1-score'] = (2 * accuracy['precision'] * accuracy['recall']) / (
                    accuracy['precision'] + accuracy['recall'])
            accuracy.fillna(0, inplace=True)

            accuracy.sort_index(inplace=True)
            if i == 0:
                if sex == 'F':
                    validation_df_female.append(['Model features'] + list(accuracy.index))
                else:
                    validation_df_male.append(['Model features'] + list(accuracy.index))

            if sex == 'F':
                validation_df_female.append([feature] + list(accuracy['f1-score']))
            else:
                validation_df_male.append([feature] + list(accuracy['f1-score']))

        if print_evolution:
            print('===========================')

    df_female_performance = pd.DataFrame(validation_df_female[1:], columns=validation_df_female[0])
    df_male_performance = pd.DataFrame(validation_df_male[1:], columns=validation_df_male[0])

    return df_female_performance, df_male_performance


def extend_features(initial_features, feature_bag, standardize=True, df_clean=None):
    """
    Given a set of initial features, extend them based on the feature bag. This function should be extended in case we
    want additional feature bags

    :param initial_features: List of features (age, height, weight and medals)
    :param feature_bag: String with the ideal feature combination
    :param standardize: Bool on whether initial features need to be standardized or not
    :param df_clean: Dataframe used in case standardization is required
    :return: List with extended features
    """

    if standardize:
        standardized_features = standardize_features(df_clean, new_features=initial_features)
    else:
        standardized_features = initial_features

    standardized_features = np.array(standardized_features)

    if feature_bag == 'featureBag_1':
        extended_features = np.append(standardized_features, [standardized_features[0] ** 2,
                                      standardized_features[1] ** 2, standardized_features[2] ** 2,
                                      standardized_features[3] ** 2,
                                      standardized_features[0] * standardized_features[1],
                                      standardized_features[0] * standardized_features[2],
                                      standardized_features[0] * standardized_features[3],
                                      standardized_features[1] * standardized_features[2],
                                      standardized_features[1] * standardized_features[3],
                                      standardized_features[2] * standardized_features[3],
                                      standardized_features[0] ** 3,
                                      standardized_features[1] ** 3, standardized_features[2] ** 3,
                                      standardized_features[3] ** 3])
    elif feature_bag == 'featureBag_2':
        extended_features = standardized_features
    elif feature_bag == 'featureBag_3':
        extended_features = np.append(standardized_features, [standardized_features[0] ** 2,
                                      standardized_features[1] ** 2, standardized_features[2] ** 2,
                                      standardized_features[3] ** 2,
                                      standardized_features[0] * standardized_features[1],
                                      standardized_features[0] * standardized_features[2],
                                      standardized_features[0] * standardized_features[3],
                                      standardized_features[1] * standardized_features[2],
                                      standardized_features[1] * standardized_features[3],
                                      standardized_features[2] * standardized_features[3]])
    elif feature_bag == 'featureBag_4':
        extended_features = np.array([standardized_features[0] ** 2,
                                      standardized_features[1] ** 2, standardized_features[2] ** 2,
                                      standardized_features[3] ** 2])
    elif feature_bag == 'featureBag_5':
        extended_features = standardized_features[1:]
    elif feature_bag == 'featureBag_6':
        extended_features = np.append(standardized_features[:-1], [standardized_features[0] ** 2,
                                      standardized_features[1] ** 2, standardized_features[2] ** 2])
    elif feature_bag == 'featureBag_7':
        extended_features = np.array([standardized_features[0] * standardized_features[1],
                                      standardized_features[0] * standardized_features[2],
                                      standardized_features[0] * standardized_features[3],
                                      standardized_features[1] * standardized_features[2],
                                      standardized_features[1] * standardized_features[3],
                                      standardized_features[2] * standardized_features[3]])
    elif feature_bag == 'featureBag_8':
        extended_features = np.array([standardized_features[0] ** 3,
                                      standardized_features[1] ** 3, standardized_features[2] ** 3,
                                      standardized_features[3] ** 3])
    elif feature_bag == 'featureBag_9':
        extended_features = np.array([standardized_features[3], standardized_features[3] ** 2,
                                      standardized_features[3] ** 3])

    return extended_features


def compute_probability_medal(sex, country, sport, age, height, weight, medals_country_dict, df_performance,
                              hyperparameter_thetas, logistic_regression_function=predict_logistic_regression,
                              df_clean=None, extend_features_function=extend_features, standardize=True):
    """
    Evaluate probability of winning a medal given a set of parameters

    :param sex: Input feature (string)
    :param country: Input feature (string)
    :param sport: Input feature (string)
    :param age: Input feature (integer)
    :param height: Input feature (integer)
    :param weight: Input feature (integer)
    :param medals_country_dict: Dictionary to map country, sex and sport to integer
    :param df_performance: (dictionary) Dataframe with performances. This will tell us which logistic model is the most appropriate
        given the sport and sex variables
    :param logistic_regression_function: Logistic function
    :return: Float (probability of winning given input features)
    """

    # Compute ideal features
    df_performance_temp = df_performance[sex]
    ideal_feature_bag = df_performance_temp.iloc[df_performance_temp[sport].argmax()]['Model features']

    medals_country = medals_country_dict[sport][country][sex]

    initial_features = [age, height, weight, medals_country]

    athlete_features = list(extend_features_function(initial_features,
                                                     feature_bag=ideal_feature_bag, df_clean=df_clean,
                                                     standardize=standardize))

    prob = logistic_regression_function(sport=sport,
                                        theta_df=hyperparameter_thetas[ideal_feature_bag][sex],
                                        new_features=athlete_features,
                                        standardize=False)

    return prob


def plot_sport_features(sport, df, x_feature='Age', y_feature='Height'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    # Focus only on the specific sport
    df = df[df.Sport == sport]

    y = df.Target_Variable.tolist()
    x_feature_list = df[x_feature].tolist()
    y_feature_list = df[y_feature].tolist()

    colors = ['#483d8b', '#cc8400']
    colors_data = [colors[int(i)] for i in y]

    ax.scatter(x_feature_list, y_feature_list, color=colors_data,
               s=100, edgecolor='black', linewidth=2, alpha=0.7)

    no_medals = mpatches.Patch(color='#483d8b', label='No Medal Won')
    medals = mpatches.Patch(color='#cc8400', label='Medal Won')

    plt.legend(handles=[no_medals, medals])

    plt.xlabel(x_feature)
    plt.ylabel(y_feature)

    ax.grid(linestyle='--', linewidth=2, color='gray', alpha=0.2)

    ax.set(xlabel=x_feature, ylabel=y_feature)

    plt.show();


def train_models(datasets_dict, classifiers_list, scoring_dict, verbose=3):
    """
    Dictionary containing all the test and train samples, for male and female

    :param datasets_dict: Dictionary containing all the test and train samples, for male and female
    :param classifiers_list: List with all the models and hyper-parameters we will evaluate using a 5-fold cross
        validation approach
    :param scoring_dict: Scores that wee will be computing. The final hyper-parameter will be chosen based on the
        F1-score
    :return: Male and female data frames with all f1-scores for each hyper-parameter values and initialized best models
        in a list
    """

    results_Female = pd.DataFrame([])
    results_Male = pd.DataFrame([])
    models_Male = []
    models_Female = []

    # Loop over both Male and Female Training Sets to fit the models specified in the classifier list
    for sex in ['F', 'M']:
        print(sex)
        print('===================')
        for name, classifier, params in classifiers_list:

            df_train_X = datasets_dict[sex]['Train'][0]
            df_train_Y = datasets_dict[sex]['Train'][1]

            if name == 'KNN':
                if sex == 'F':
                    clf = GridSearchCV(estimator=classifier,
                                       param_grid=params['n_neighbors_female'],
                                       scoring=scoring_dict,
                                       cv=None,
                                       n_jobs=-1,
                                       refit='weighted_F1',
                                       verbose=verbose)
                else:
                    clf = GridSearchCV(estimator=classifier,
                                       param_grid=params['n_neighbors_male'],
                                       scoring=scoring_dict,
                                       cv=None,
                                       n_jobs=-1,
                                       refit='weighted_F1',
                                       verbose=verbose)
            else:
                # Initialize class (GridSearchCV) over which we will
                # perform the hyperparameter search for each of the above mentioned models
                # Model applied on training data and evaluated over test data
                clf = GridSearchCV(estimator=classifier,
                                   # Paramater of each models (as required by the sklearn function of each model)
                                   param_grid=params,
                                   # Set of evaluation metrics over which the model will be evaluated
                                   scoring=scoring_dict,
                                   # Apply 5-fold cross validation approach
                                   cv=None,
                                   # Apply paralisation over all possible processors
                                   n_jobs=-1,
                                   # Refit an estimator using the best found parameters on the whole dataset.
                                   # This is required when more than 1 scoring approaches are given (as in our case)
                                   refit='weighted_F1',
                                   verbose=verbose)
            print('Model:', name)

            # Fit models using the previously initialized class (GridSearchCV)
            fit = clf.fit(df_train_X, df_train_Y)

            # Keep only the results that will help us choose the best model to predict our ideal sport
            search = pd.DataFrame.from_dict(fit.cv_results_)[['params', 'mean_test_accuracy',
                                                              'mean_test_weighted_precision',
                                                              'mean_test_weighted_recall',
                                                              'mean_test_weighted_F1']]

            # Include column with the model name and remove mean_test from the columns
            search['Model'] = name
            search.columns = search.columns.str.replace('mean_test_', '')

            # Given we have computed our "real" predicition model we will now go and classify
            # using a random approach to see whether our model predicts better than the random approach
            dum_class = DummyClassifier(strategy='uniform', random_state=len(df_train_Y.unique()))

            # Compute metrics over these as well
            dum = cross_validate(dum_class, df_train_X, df_train_Y, cv=5, scoring=scoring_dict)
            # Keep relevant fit output parameters
            dum = pd.DataFrame.from_dict(dum).drop(columns=["fit_time", "score_time"])
            dum['Model'] = name

            # Keep the average over all cross-validation k-fold values
            dum = dum.assign(**dum.mean()).iloc[[0]]

            # Rename columns to later join with the search dataframe
            dum.columns = dum.columns.str.replace('test_', 'dummy_')

            search = pd.merge(search, dum, how='left', on=['Model'])

            # Include the best estimator of the previous model into the model list
            if sex == 'F':
                models_Female.append((name, fit.best_estimator_))
                results_Female = results_Female.append(search, ignore_index=True)
            else:
                models_Male.append((name, fit.best_estimator_))
                results_Male = results_Male.append(search, ignore_index=True)

    return models_Female, results_Female, models_Male, results_Male


def evaluate_models(datasets, models, le):
    """
    Evaluate the best models (stored in the dictionary models) over the test data sets (stored in the dictionary
    datasets)

    :param datasets: Dictionary with all data sets required
    :param models: Best hyper-parameter models
    :param le: LabelEncoder class
    :return: A dataframe with the all metrics to evaulate the models at sex and sport level
    """

    df_final = pd.DataFrame([])
    for sex in ['F', 'M']:
        print(sex)
        print('===============')
        for name, model in models[sex]:
            df_test_X = datasets[sex]['Test'][0]
            df_test_Y = datasets[sex]['Test'][1]

            # Given the best model predict based on the X features and store results in Y_pred
            Y_pred = model.predict(df_test_X)
            # Convert label back into sport name
            Y_pred = le.inverse_transform(Y_pred)

            # Compute the actual sport labels and see how good our model was able to predict at sport and overall level
            Y_actual = df_test_Y
            Y_actual = le.inverse_transform(Y_actual)

            # Generate report between the predicted and the actual results
            temp_results = classification_report(Y_actual, Y_pred, output_dict=True)

            # Leave out results of 'accuracy', 'macro avg' and 'weighted avg' (we will compute this at an overall level
            # later on)
            temp_results = {key: temp_results[key] for key in temp_results if
                            key not in ['accuracy', 'macro avg', 'weighted avg']}

            # Compute precision, recall and f1 for the overall model and sex
            precision, recall, f_score, support = precision_recall_fscore_support(Y_actual, Y_pred, average="weighted")

            # Generate dataframe with final results and store everything in df_final
            temp_results['overall'] = {'precision': precision, 'recall': recall, 'f1-score': f_score,
                                       'support': support}
            df_results = pd.DataFrame.from_dict(temp_results).T
            df_results['Sex'] = sex
            df_results['Model'] = name
            df_final = df_final.append(df_results)

    return df_final


def predict_ideal_sport_and_prob(df, models, le, medals_country_dict, df_clean, df_performance, hyperparameter_thetas):
    """
    Given a data set with correct input features, and the champion model, return the ideal sport for each athlete.


    :return: Data set with ideal sport and chances of winning a medal
    """
    medals = []
    for index, row in df.iterrows():
        temp = []
        for ele in medals_country_dict:
            try:
                temp.append(medals_country_dict[ele][row[4]][row[5]])
            except:
                continue
        medals.append(max(temp))

    df['Medals_per_country'] = medals

    athlete_sex = df.Sex.tolist()

    i = -1
    y_label = []
    p_medal = []
    for index, row in df.iterrows():
        std_features = standardize_features(df_clean,
                                            features=['Age', 'Height', 'Weight', 'Medals_per_country'],
                                            new_features=[int(row[1]), int(row[2]), int(row[3]), int(row[6])])

        temp_df = pd.DataFrame(data=[[int(row[0])] + std_features],
                               columns=['Year', 'Age', 'Height', 'Weight', 'Medals_per_country'])

        i += 1
        # Random Forest!
        model = models[athlete_sex[i]][1][1]
        y_predict = model.predict(temp_df)
        sport = le.inverse_transform(y_predict)[0]
        y_label.append(sport)

        #medals_country = medals_country_dict[sport][int(row[4])][int(row[5])]
        p_medal.append(compute_probability_medal(sex=row[5], country=row[4], sport=sport, age=int(row[1]),
                                                 height=int(row[2]), weight=int(row[3]),
                                                 medals_country_dict=medals_country_dict,
                                                 df_performance=df_performance,
                                                 hyperparameter_thetas=hyperparameter_thetas, df_clean=df_clean,
                                                 standardize=True))

    df['Sport_predict'] = y_label
    df['P_Medal'] = p_medal
    df.drop('Medals_per_country', axis=1, inplace=True)
    return df