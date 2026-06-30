import torch
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_metrics(confusion_vector, total, test_loss, num_batches):
    tn = int(confusion_vector[0][0])
    tp = int(confusion_vector[1][1])
    fp = int(confusion_vector[0][1])
    fn = int(confusion_vector[1][0])

    recall = max(0, tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = max(0, tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = max(0, tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (recall * precision) / (precision + recall)

    metrics = {
        'loss': test_loss / num_batches,
        'accuracy': 100.0 * accuracy,
        'balanced_accuracy': 100.0 * (recall + specificity) / 2,
        'recall': recall,
        'precision': precision,
        'F1-score': f1_score,
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }
    return metrics


def print_metrics(metrics, title='Test'):
    print(f"\n{title} metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['F1-score']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {metrics['confusion_matrix']}")


def test(data_loader, model, criterion, epoch_logger=None, title='Test'):
    # Switch to test mode
    model.eval()
    test_loss, total, confusion_vector = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):

            inputs = inputs.to(device)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs.cpu(), targets.unsqueeze(1).float())
            # Calculate loss
            test_loss += loss.item()
            # Make predictions
            predicted = outputs >= 0.5
            total += targets.size(0)
            # Extract evaluation metrics
            confusion_vector += confusion_matrix(targets.cpu(), predicted.cpu(), labels=[0, 1])

    metrics = compute_metrics(confusion_vector, total, test_loss, batch_idx + 1)
    print_metrics(metrics, title=title)

    if epoch_logger is not None:
        epoch_logger.log({
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'F1-score': metrics['F1-score'],
            'confusion_matrix': str(metrics['confusion_matrix']),
        })

    return metrics
