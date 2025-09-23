import argparse, os, joblib, mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.model_utils import load_data, get_feature_columns, build_preprocessor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=str, default='data/train.csv', help="Path to training data")
    p.add_argument('--test', type=str, default='data/test.csv', help="Path to test data")
    p.add_argument('--out', type=str, default='models/model.pkl', help="Output model path")
    p.add_argument('--n-estimators', type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load datasets
    train_df = load_data(args.train)
    test_df = load_data(args.test)

    num_cols, cat_cols = get_feature_columns(train_df)
    target_col = 'target'

    # Handle missing numeric values and cast to float
    for col in num_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val).astype('float64')
        test_df[col] = test_df[col].fillna(median_val).astype('float64')

    # Impute missing categorical values
    for col in cat_cols:
        mode_val = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)

    # Separate features and target
    X_train, y_train = train_df[num_cols + cat_cols], train_df[target_col]
    X_test, y_test = test_df[num_cols + cat_cols], test_df[target_col]

    # Print target distribution
    print("Train target distribution:\n", y_train.value_counts())
    print("Test target distribution:\n", y_test.value_counts())

    # Build preprocessor and pipeline with class weighting
    preproc = build_preprocessor(num_cols, cat_cols)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        class_weight='balanced'  # handle class imbalance
    )
    pipe = Pipeline([('preproc', preproc), ('clf', clf)])

    # MLflow autolog
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

        # ROC AUC
        if len(y_test.unique()) > 1:
            probas = pipe.predict_proba(X_test)[:,1]
            print("ROC AUC:", roc_auc_score(y_test, probas))

        # Save trained pipeline
        joblib.dump(pipe, args.out)
        print(f"Model saved to {args.out}")

if __name__ == "__main__":
    main()
