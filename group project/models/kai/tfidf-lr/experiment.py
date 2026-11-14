"""Notebook cell for the CPU baseline. Variables come from the shared data cell."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

held_out = read_snapshot("test")
baseline_rows = []
for variant, fit_data in [("same_training_subset", train_df), ("full_training_pool", pool)]:
    estimator = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=7))
    estimator.fit(fit_data.text_clean, fit_data.Sentiment)
    predicted = estimator.predict(held_out.text_clean)
    scores = classification_report(held_out.Sentiment, predicted, labels=LABEL_IDS,
                                   target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    write_json(RUN_DIR / f"tfidf_{variant}_metrics.json", scores)
    held_out.assign(predicted=predicted)[["row_id", "Sentiment", "predicted"]].to_csv(
        RUN_DIR / f"tfidf_{variant}_predictions.csv", index=False)
    baseline_rows.append({"variant": variant, "train_n": len(fit_data), "test_n": len(held_out),
                          "accuracy": accuracy_score(held_out.Sentiment, predicted),
                          "macro_f1": scores["macro avg"]["f1-score"]})
baseline_table = pd.DataFrame(baseline_rows)
display(baseline_table)
baseline_table.to_csv(RUN_DIR / "tfidf_comparison.csv", index=False)
print("结果目录", RUN_DIR)
