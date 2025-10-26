# Inference Guide - Understanding Model Performance

## What is This Script?

The `inference.py` script loads your trained model and runs it on test images to evaluate its performance. It provides detailed metrics and visualizations to help you understand how well your model is working.

## Key Concepts

### 1. **IoU (Intersection over Union)**
- **What it is**: Measures how well predicted boxes overlap with ground truth boxes
- **Range**: 0 to 1 (higher is better)
- **Interpretation**:
  - 0.7-1.0: Excellent overlap
  - 0.5-0.7: Good overlap
  - 0.3-0.5: Moderate overlap
  - 0.0-0.3: Poor overlap

### 2. **Precision**
- **What it is**: Out of all predictions the model made, how many were correct?
- **Formula**: True Positives / (True Positives + False Positives)
- **Interpretation**: Higher precision = fewer false positives
- **Example**: Precision of 0.8 means 80% of predictions are correct

### 3. **Recall**
- **What it is**: Out of all ground truth objects, how many did the model find?
- **Formula**: True Positives / (True Positives + False Negatives)
- **Interpretation**: Higher recall = fewer missed objects
- **Example**: Recall of 0.8 means the model found 80% of all objects

### 4. **F1 Score**
- **What it is**: Harmonic mean of precision and recall
- **Interpretation**: Balanced measure considering both precision and recall
- **When to use**: When you need a single number to compare models

### 5. **True Positives, False Positives, False Negatives**
- **True Positive (TP)**: Correctly detected object
- **False Positive (FP)**: Predicted object that doesn't exist (false alarm)
- **False Negative (FN)**: Missed a real object

## How to Run Inference

### Basic Usage

```bash
python inference.py --model models/model_20241019_123456.pth
```

### With Custom Options

```bash
# Process specific number of samples
python inference.py --model models/model.pth --num_samples 50

# Skip visualizations (faster)
python inference.py --model models/model.pth --no_viz

# Custom output directory
python inference.py --model models/model.pth --output my_results
```

## Understanding the Output

### Visualizations

The script creates images showing:
- **Green boxes**: Ground truth (actual objects in the image)
- **Red boxes**: Model predictions
- **Scores**: Confidence level for each prediction (0-1)

**What to look for:**
1. Do red boxes match green boxes? (Good detection)
2. Are there red boxes with no green boxes? (False positives)
3. Are there green boxes with no red boxes? (Missed objects)
4. Are box sizes similar? (Good localization)

### Metrics Table

The table shows:
- **Total True Positives**: Number of correct predictions
- **Total False Positives**: Number of incorrect predictions
- **Total False Negatives**: Number of missed objects
- **Mean IoU**: Average box overlap quality
- **Precision**: Fraction of predictions that are correct
- **Recall**: Fraction of objects found
- **F1 Score**: Overall balanced performance metric

### JSON Results

Saved to `results.json`, containing:
- All metrics as numbers
- Model path used
- Timestamp of inference run
- Number of samples processed

## How to Interpret Results

### Good Model Performance

✅ **High Precision (>0.7)**: Model makes few mistakes
✅ **High Recall (>0.7)**: Model finds most objects
✅ **High Mean IoU (>0.5)**: Boxes are well positioned
✅ **F1 Score >0.7**: Overall good performance

### Areas for Improvement

⚠️ **Low Precision**: Model has many false positives
   - Solution: Increase confidence threshold, add training data with negative examples

⚠️ **Low Recall**: Model misses many objects
   - Solution: Improve detection head, decrease confidence threshold, add more training data

⚠️ **Low IoU**: Model finds objects but boxes are inaccurate
   - Solution: Improve box regression loss, add more training data

### Real-World Interpretation

**Use Case: Surgical Tool Detection**
- **Priority: Safety** → Need HIGH precision (few false alarms)
- **Priority: Completeness** → Need HIGH recall (don't miss tools)
- **Priority: Accuracy** → Need HIGH IoU (know exact tool position)

## Common Issues and Solutions

### Issue: Very high precision but low recall
**Problem**: Model is too conservative, only makes predictions when very confident
**Solution**: Lower confidence threshold, improve training data diversity

### Issue: Very high recall but low precision
**Problem**: Model is making too many predictions
**Solution**: Raise confidence threshold, add negative examples to training

### Issue: Low IoU scores
**Problem**: Model finds objects but boxes are offset
**Solution**: Improve box regression, add more box examples to training

## Best Practices

1. **Run inference after training**: Always validate your model
2. **Use validation set**: Don't just test on training data
3. **Analyze failures**: Look at images with poor IoU to understand patterns
4. **Track metrics**: Compare metrics across different training runs
5. **Visualize randomly**: Check a mix of good and bad predictions

## Next Steps

After running inference:

1. **If metrics are good**: Deploy model or test on new data
2. **If metrics need improvement**: 
   - Collect more training data
   - Tune hyperparameters
   - Try different architectures
   - Adjust loss function weights

3. **Compare with baseline**: Use these metrics to track progress
4. **Iterate**: Model improvement is an iterative process

## Example Output Interpretation

```
Mean IoU: 0.65        → Good box localization
Precision: 0.82       → Model makes few mistakes
Recall: 0.71          → Model finds most objects
F1 Score: 0.76        → Overall solid performance

Interpretation: Model is performing well with good balance between
precision and recall. IoU indicates boxes are reasonably well positioned.
Consider fine-tuning if you need higher recall (more complete detection).
```

## Questions?

- **What's a good score?**: Depends on your use case, but generally:
  - mAP >0.5: Acceptable
  - mAP >0.7: Good
  - mAP >0.9: Excellent
  
- **How many samples to test?**: At least 100-200 for reliable metrics
  
- **When to retrain?**: When precision or recall drops below 0.6-0.7 for your use case
