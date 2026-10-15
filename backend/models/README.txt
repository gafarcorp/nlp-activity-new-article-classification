Put the trained model files from the notebook in this folder:

  tfidf_vectorizer.joblib      (needed by Logistic Regression and Random Forest)
  best_model_logreg.joblib
  random_forest.joblib
  bert_ag_news/                (folder written by trainer.save_model + tokenizer.save_pretrained)

Any model whose file is missing is shown as "not loaded" in the app; the others still work.
These files are large and are ignored by git (see .gitignore).
