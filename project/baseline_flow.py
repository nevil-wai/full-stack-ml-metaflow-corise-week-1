from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
# labeling_function = lambda row: 0
def labeling_function(row):
    """
    A function to derive labels from the user's review data.
    This could use many variables, or just one. 
    In supervised learning scenarios, this is a very important part of determining what the machine learns!
   
    A subset of variables in the e-commerce fashion review dataset to consider for labels you could use in ML tasks include:
        # rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
        # recommended_ind: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
        # positive_feedback_count: Positive Integer documenting the number of other customers who found this review positive.

    In this case, we are doing sentiment analysis. 
    To keep things simple, we use the rating only, and return a binary positive or negative sentiment score based on an arbitrarty cutoff. 
    """
    # TODO: Add your logic for the labelling function here
    # It is up to you on what value to choose as the cut off point for the postive class
    # A good value to start would be 4
    # This function should return either a 0 or 1 depending on the rating of a particular row
    return 1 if row.rating > 3 else 0

def pre_process_review(review): 

    import re
    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()
    
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(review))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    return document


class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        import nltk
        from nltk.corpus import stopwords
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        
        # Convert review to TFIDF vector
        tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        reviews = tfidfconverter.fit_transform(_has_review_df['review_text']).toarray()
        labels = _has_review_df.apply(labeling_function, axis=1)

        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        # self.df = pd.DataFrame({'label': labels, **_has_review_df['review_vec']})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(reviews, labels, test_size=0.2, random_state=0)
        print(f'num of rows in train set: {self.X_train.shape[0]}')
        print(f'num of rows in validation set: {self.X_test.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        self.clf = RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2, random_state=0
        )
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

        self.base_acc = 0.0
        self.base_rocauc = 0.0
        self.base_acc = accuracy_score(self.y_test, self.y_pred)
        self.base_rocauc = roc_auc_score(self.y_test, self.y_pred)

        self.df = pd.DataFrame({'y_test' : self.y_test, 'y_pred': self.y_pred})
        self.next(self.end)
        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        import pandas as pd 

        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.
        fp_df = self.df[(self.df['y_test']==0)&(self.df['y_pred']==1)]
        current.card.append(Markdown("## False Positive Predictions"))
        # TODO: display the false_positives dataframe using metaflow.cards
        current.card.append(Table.from_dataframe(fp_df))
        # Documentation: https://docs.metaflow.org/api/cards#table
        
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1. 
        fp_df = self.df[(self.df['y_test']==1)&(self.df['y_pred']==0)]
        # TODO: display the false_negatives dataframe using metaflow.cards
        current.card.append(Table.from_dataframe(fp_df))

if __name__ == '__main__':
    BaselineNLPFlow()
