import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(path, predictions, labels, num_classes):
    
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[f'C{i}' for i in range(num_classes)])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=True)

    plt.xticks(rotation=45, ha='right')
    plt.title('21-Class Confusion Matrix')
    plt.tight_layout()
    # plt.show()
    plt.savefig(path, dpi=150)