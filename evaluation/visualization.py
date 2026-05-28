import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion_matrix_viz(y_true,
                         y_pred,
                         class_names=None):

    cm = confusion_matrix(y_true, y_pred)
    class_names = class_names
    
    plt.figure(figsize=(4, 3))

    if class_names == None:
        sns.heatmap(
            cm, 
            annot=True,            
            fmt='d',              
            cmap='Blues',       
            square=True,          
            cbar_kws={'label': 'Number of Predictions'} 
        )

    else:
        sns.heatmap(
            cm, 
            annot=True,            
            fmt='d',              
            cmap='Blues',       
            xticklabels=class_names, 
            yticklabels=class_names,
            square=True,          
            cbar_kws={'label': 'Number of Predictions'} 
        )
    
    plt.title('Confusion Matrix Heatmap', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.show()