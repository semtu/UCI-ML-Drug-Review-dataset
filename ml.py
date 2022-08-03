import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class data_visualization(object):
    """
    Class for data_visualization functions
    """

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.df = self.data_preprocessing()

    def load_dataset(self):
        """
        Load train and test dataset

        Returns:
            Tuple of train and test data objects
        """
        try:
            train = pd.read_csv(os.path.join(
                self.BASE_DIR, "drugsComTrain_raw.csv"))
            test = pd.read_csv(os.path.join(
                self.BASE_DIR, "drugsComTest_raw.csv"))
        except (FileNotFoundError, OSError):
            print("Error opening files")
        return train, test

    def data_preprocessing(self):
        """
        Preprocesses the data
            1. Concatenate train and test dataframes
            2. Drop null entries
            3. Compounds the patients ratings (positive if ratings is greater than or equal to 5
                or negative if ratings is less than 5)
            4. extracts the year feature for the date column
            5. Removes html hex strings found in the text using regular expressions

        Returns:
            dataframe object
        """
        df = pd.concat([self.load_dataset()[0], self.load_dataset()[1]])
        # df.info() # 4 categorical columns and 3 numeric columns

        df.review.replace(
            [r"&#039;", r"&#45", r"&#39", r"&amp"],
            ["'", "-", "", "and"],
            regex=True,
            inplace=True,
        )  # getting rid of html symbols
        # normalizing to lower case
        df.review = df.review.apply(lambda x: x.lower())

        df.isnull().describe()  # condition column is seen to contain null entries
        df[
            df["condition"].isnull()
        ]  # Specifically, it has 1194 null entries, these rows will be dropped
        df = df[
            df["condition"].notna()
        ]  # dropping null entries by indexing rows where value is not null

        # ratings >= 5 are classed positive
        # ratings < 5: Negative
        df["compound_rating"] = df["rating"].apply(
            lambda x: "Positive" if x >= 5 else "Negative"
        )

        type(
            df["date"].iloc[0]
        )  # the date column has an object type, we can convert to datetime objects
        df["date"] = pd.to_datetime(df["date"])
        df["Year"] = df["date"].apply(lambda year: year.year)
        return df

    def graph_generator(self, feature, data, order, file_name):
        """
        Generates bar plots

        Parameter:
            feature: the data column to create the plot for
            data
            order: sets the order in which data is represented on the plot
            file_name: file path
        """
        plt.figure(figsize=(15, 8))
        plt.title(file_name[:-3])
        sns.set(style="whitegrid", font_scale=1.5)
        ax1 = sns.countplot(
            x=feature,
            data=data,
            palette="viridis",
            order=order.value_counts().iloc[:10].index,
        )
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.BASE_DIR, file_name))
        return

    def results(self):
        """
        This function visualizes the data and stores the plots in {results/}
        """
        df = self.df

        # Top 10 drugs that received the most ratings - This shows the drugs that patients used the most initially
        self.graph_generator(
            "drugName",
            df,
            df.drugName,
            "results/Top_10_drugs_that_received_the_most_ratings.png",
        )

        # Similarly, lets look at top 10 conditions patients suffered from
        self.graph_generator(
            "condition",
            df,
            df.condition,
            "results/Top_10_conditions_that_patients_suffered_from.png",
        )

        # Birth Control is significantly higher than others
        print(
            f"Birth control accounts for {df.condition.value_counts().iloc[0]/len(df)*100:.1f} % of all conditions"
        )

        # A closer look at birth control condition.
        # Let us know the most common drugs different patients have used for birth control
        self.graph_generator(
            "drugName",
            df[df.condition == "Birth Control"],
            df[df.condition == "Birth Control"].drugName,
            "results/most_common_drugs_used_for_birth_control.png",
        )

        # It turns out most patients perferred Etonogestrel for birth control. But what exactly
        # are their sentiments after making use of the drug?

        # Lets understand how patients rated these drugs using a compound scale
        # Overall, how did patients review the drugs?
        compound = df.compound_rating.value_counts().to_frame().T
        labels = compound.columns.to_list()
        sizes = [compound[labels[0]].iloc[0], compound[labels[1]].iloc[0]]
        explode = (0.1, 0)
        fig1, ax1 = plt.subplots(figsize=(15, 8))
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax1.axis("equal")
        plt.title("patients_review_of_drugs")
        plt.tight_layout()
        plt.savefig(os.path.join(self.BASE_DIR,
                    "results/patients_review_of_drugs.png"))

        # The reviews by patients were largely positive with over 70% positive ratings of the total

        # Now that we know overall patients sentiments, let us look at the top 10 drugs with most positive sentiments
        self.graph_generator(
            "drugName",
            df[df.compound_rating == "Positive"],
            df[df.compound_rating == "Positive"].drugName,
            "results/Top_10_drugs_with_the_most_positve_sentiment.png",
        )
        # Levonorgestrel drug has the most positive reviews

        # Similarly we can take a look at the drug with the most negative sentiments
        self.graph_generator(
            "drugName",
            df[df.compound_rating == "Negative"],
            df[df.compound_rating == "Negative"].drugName,
            "results/Top_10_drugs_with_the_most_negative_sentiment.png",
        )
        # Etonogestrel drug has the most negative reviews

        # We know that birth control is the most prevalent condition and patients with this condition
        # made use of Etonogestrel the most.
        # Let us see how effective the drug was for birth control
        birth_control_plot = (
            df[
                (df["condition"] == "Birth Control")
                & (df["drugName"] == "Etonogestrel")
            ]["compound_rating"]
            .value_counts()
            .plot(kind="barh")
        )
        birth_control_plot.set_title(
            "patients_review_of_Etonogestrel_for_birth_control"
        )
        fig = birth_control_plot.get_figure()
        fig.savefig(
            os.path.join(
                self.BASE_DIR,
                "results/patients_review_of_Etonogestrel_for_birth_control.png",
            )
        )
        # A little above 50% of patients were positively satisfied using the drug for birth control. However, the number of
        # negative reviews is not far behind, more tests should be conducted to ascertain the effectiveness of the drug.

        # What information can we get from date column?
        # What year were the most reviews given and are there any significant relationship between the year and patient's sentiment?
        plt.figure(figsize=(15, 8))
        plt.title("year_most_reviews_given")
        sns.set(style="whitegrid", font_scale=1.5)
        plot1 = sns.countplot(
            x="Year", data=df, palette="viridis", hue="compound_rating"
        )
        plot1.set_xticklabels(plot1.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.BASE_DIR,
                    "results/year_most_reviews_given.png"))
        # 2016
        # There is a drastic increase in negative sentiments between the year 2014 and 2015. Why is that so?
        return


class models(data_visualization):
    """
    This is a class for the model functions. Inherits processed dataframe from data_visualization parent class

    Attributes:
        ''
    """

    def __init__(self):
        """
        The constructor for the class
        """
        super().__init__()  # initialize attributes of data_visualization class
        self.X, self.review_X, self.y = self.feature_engineering()
        self.df = data_visualization().df

    def text_process(self, text):
        """
        Takes a text string as input and carries out the following actions:
            1. Eliminates any punctuation
            2. Eliminate any stopwords.
            3. returns a list of the text that has been cleaned

        Parameter:
            text : str

        Returns:
            list of clean texts
        """
        StopWords = stopwords.words("english")
        punctuation = string.punctuation
        no_punctuation = [char for char in text if char not in punctuation]
        no_punctuation = "".join(no_punctuation)
        return [
            word for word in no_punctuation.split() if word.lower() not in StopWords
        ]

    def feature_engineering(self):
        """
        Generates features that can be used during model training. Features generated are:
            1. word count: number of words in text
            2. number of characters in text
            3. the average word length for each text
            4. word count after cleaning the text. Cleaning involved removing punctuation and stop words

        Returns:
            Features and label dataframe objects
        """
        from sklearn.preprocessing import LabelEncoder

        self.df["word_count"] = self.df.review.apply(
            lambda x: len(str(x).split()))
        self.df["num_of_chars"] = self.df.review.apply(lambda x: len(str(x)))
        self.df["clean_review"] = self.df["review"].apply(
            lambda x: self.text_process(x)
        )
        self.df["mean_word_length"] = self.df.review.apply(
            lambda x: np.mean([len(word) for word in str(x).split()])
        )
        self.df["word_count_after_cleaning"] = self.df.clean_review.apply(
            lambda x: len(x)
        )

        cache_label_encoder = dict()
        for feature in [
            "drugName",
            "condition",
        ]:  # To convert drugName and condition columns to numerical variables
            cache_label_encoder[feature] = LabelEncoder()
            self.df[feature] = cache_label_encoder[feature].fit_transform(
                self.df[feature]
            )

        X = self.df[
            [
                "drugName",
                "condition",
                "usefulCount",
                "Year",
                "word_count",
                "num_of_chars",
                "mean_word_length",
                "word_count_after_cleaning",
            ]
        ]
        review_feature = self.df["review"]
        y = self.df["compound_rating"]
        return X, review_feature, y

    def train_test_split(self):
        """
        Split data into train and test. We want to avoid training the model with 100 % data to avoid overfitting

        Returns:
            Tuple of split train and test data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def xgboost_model(self):
        """
        This function trains the XGBclassifier
        """
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test = self.train_test_split()

        xgb_model = XGBClassifier(
            random_state=42, seed=2, colsample_bytree=0.6, subsample=0.9
        )
        xgb_model.fit(X_train, y_train)

        xgb_predictions = xgb_model.predict(X_test)
        print(
            f"Xgboost classification report: {classification_report(y_test,xgb_predictions)}"
        )

        return

    def lgbm_model(self):
        """
        This function trains the LGBMClassifier
        """
        from lightgbm import LGBMClassifier

        X_train, X_test, y_train, y_test = self.train_test_split()
        lgbm_model = LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=100000,
            subsample=0.9,
            max_depth=7,
            min_split_gain=0.01,
            min_child_weight=2,
        )
        lgbm_model.fit(X_train, y_train)

        # Predictions
        lgbm_predictions = lgbm_model.predict(X_test)
        print(
            f"LGBM classification report: {classification_report(y_test,lgbm_predictions)}"
        )
        return

    def multinomialNB_model(self):
        """
        This function uses only the review features to classify the compound ratings in a structured pipeline
        1. Count vecorization: converts the texts to integer tokens counts
        2. TF-IDF Normalization: The integer tokens are normalized using TF-IDF scores
        3. Multinomial NB: Uses a Naive Bayes classifier to classify the discrete features of the reviews
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline

        X_train, X_test, y_train, y_test = train_test_split(
            self.review_X, self.y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline(
            [
                (
                    "bow",
                    CountVectorizer(analyzer=self.text_process),
                ),  # Count vectorizer converts the texts to integer tokens
                (
                    "tfidf",
                    TfidfTransformer(),
                ),  # The integer tokens are normalized using TF-IDF scores
                (
                    "classifier",
                    MultinomialNB(),
                ),  # train the normalized vectors with a Multinomial Naive Bayes classifier
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        print(
            f"MultinomialNB classification report: {classification_report(predictions,y_test)}"
        )
        return


if __name__ == "__main__":
    visuals = (
        data_visualization().results()
    )  # This method generates the plots for data visualization (stored in results/)

    model_obj = models()  # initialize model object
    model_obj.xgboost_model()  # call xgb method
    model_obj.lgbm_model()  # call lgbm model
    model_obj.multinomialNB_model()  # run pipeline model

"""
Summary:

LGBM classifier reports the best accuracy of 83%. XGBoost and MultinomialNB both have an accuracy of 77%.
MultinomialNB has a 93% recall call which shows a large number of false negatives.

The models have a good performance but can be further improved by
1. More feature engineering: We can extract features like parts of speech, abstraction [1], paragraphs,
    number of sentences in each text, sentimental biases in patients review
2. Hyperparameter tuning using methods like grid search, cross validation....
3. Use different discrete classifier algorithms

References
[1] Johnson-Grey KM, Boghrati R, Wakslak CJ, Dehghani M. Measuring Abstract Mind-Sets Through Syntax:
    Automating the Linguistic Category Model. Social Psychological and Personality Science. 2020;11(2):217-225.
    doi:10.1177/1948550619848004
"""
