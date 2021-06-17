from sklearn.model_selection import train_test_split

def split_train_test(df):
    print("received df in split train test ",+df.head())
    X_df = df.drop('record_type',axis=1)
    y_recordtype = df['record_type']
    # Split the data into a training and test set.
    Xlr, Xtestlr, ylr, ytestlr = train_test_split(X_df,
                                                  y_recordtype, test_size=0.1,random_state=42)
    return Xlr, Xtestlr, ylr, ytestlr

