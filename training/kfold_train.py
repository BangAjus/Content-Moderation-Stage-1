from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score
import numpy as np

def stratified_kfold_train_val(n_splits,
                               thresholds,
                               model,
                               x_train,
                               y_train):

    skf = StratifiedKFold(n_splits=n_splits, 
                          shuffle=True, 
                          random_state=42)

    fold_recall = []
    fold_precision = []
    threshold = 0.35
    
    print(f"Starting 5-Fold Stratified Cross-Validation...\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train,
                                                          y_train), 1):
        
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        model.fit(x_train_fold, 
                  y_train_fold)
        
        try:

            test_probabilities = model.predict_proba(x_val_fold)[:, 1]
            predictions = (test_probabilities >= threshold).astype(int)
            print("sss")
            custom_recall = recall_score(y_val_fold, predictions)
            fold_recall.append(custom_recall)
            
            custom_precision = precision_score(y_val_fold, predictions)
            fold_precision.append(custom_precision)
            
            print(f"--- Performance Analysis at Threshold ({threshold:.2f}) ---")
            print(f"Custom Precision Score: {custom_precision:.4f} (When flagged positive, accuracy is {custom_precision:.2%})")
            print(f"Custom Recall Score:    {custom_recall:.4f} (Captured {custom_recall:.2%} of all true positive cases)")

        except:

            print("model cannot use threshold")
            return
    
    print("\n--- Final Cross-Validation Metrics Summary ---")
    
    print(f"All Fold Recall Scores: {[round(s, 4) for s in fold_recall]}")
    print(f"Mean Recall Vector Score: {np.mean(fold_recall):.4f}")
    print(f"Standard Deviation Variance:  {np.std(fold_recall):.4f}")
    
    print(f"All Fold Precision Scores: {[round(s, 4) for s in fold_precision]}")
    print(f"Mean Precision Vector Score: {np.mean(fold_precision):.4f}")
    print(f"Standard Deviation Variance:  {np.std(fold_precision):.4f}")