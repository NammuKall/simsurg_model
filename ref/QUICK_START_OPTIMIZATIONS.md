# Quick Start: Model Optimizations

## Immediate Actions (This Week)

### 1. Add Data Augmentation (2-3 hours)
**File:** `src/data/coco_data_loader.py`

Add to `ToTensor` class or create new augmentation transforms:
- Random horizontal flip (50%)
- Color jitter (brightness, contrast, saturation)
- Random scaling (0.9x to 1.1x)

**Expected Gain:** +10-15% mAP

---

### 2. Add Learning Rate Scheduler (1 hour)
**File:** `main.py` in `setup_device_and_model()`

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# After creating optimizer:
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
# Or use OneCycleLR for faster convergence
```

**Expected Gain:** +5-10% final accuracy

---

### 3. Add NMS to Post-Processing (2 hours)
**File:** `src/models.py` in `postprocess_predictions()`

```python
from torchvision.ops import nms

# After filtering by score threshold:
keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
filtered_boxes = filtered_boxes[keep]
filtered_scores = filtered_scores[keep]
filtered_labels = filtered_labels[keep]
```

**Expected Gain:** +5-10% precision

---

### 4. Add Gradient Clipping (30 minutes)
**File:** `src/training.py` in `train_epoch()`

```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Expected Gain:** More stable training

---

## Next Week Actions

### 5. Implement Focal Loss (4-6 hours)
**File:** Create `src/training/losses.py`

Implement focal loss for classification:
- Alpha: 0.25
- Gamma: 2.0
- Replace current cross-entropy

**Expected Gain:** +5-10% mAP

---

### 6. Add Mixed Precision Training (2 hours)
**File:** `src/training.py`

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(images, targets)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Expected Gain:** 2x faster training, same accuracy

---

## Model Upgrade Path

### Option 1: Faster R-CNN (Easiest, Best Accuracy)
**Time:** 1-2 days
**File:** Create `src/models/faster_rcnn.py`

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def create_faster_rcnn(num_classes=2):
    model = fasterrcnn_resnet50_fpn(
        weights='DEFAULT',
        num_classes=num_classes
    )
    return model
```

**Expected Gain:** +20-30% mAP

---

### Option 2: YOLOv8 (Best Performance)
**Time:** 2-3 days
**Requires:** `pip install ultralytics`

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s, yolov8m, yolov8l, yolov8x
model.train(data='path/to/coco.yaml', epochs=10)
```

**Expected Gain:** +25-35% mAP, faster inference

---

## Configuration Changes

### Update `.env` file:
```bash
# Add these new options:
USE_AUGMENTATION=true
USE_MIXED_PRECISION=true
USE_SCHEDULER=cosine
GRADIENT_CLIP_NORM=1.0
MODEL_TYPE=efficientdet  # or faster_rcnn, yolo
```

---

## Testing Checklist

After each optimization:
- [ ] Training runs without errors
- [ ] Validation loss decreases
- [ ] mAP improves (or stays same)
- [ ] Inference speed acceptable
- [ ] Memory usage acceptable

---

## Performance Monitoring

Track these metrics in W&B:
- Training/validation loss
- mAP (mean Average Precision)
- Precision/Recall
- F1 Score
- Inference time (ms per image)
- GPU memory usage

---

## Rollback Plan

If optimizations cause issues:
1. Git commit before each change
2. Test on small subset first
3. Keep old code commented out
4. Use feature flags in config

---

## Expected Timeline

**Week 1:**
- Day 1-2: Data augmentation + LR scheduler
- Day 3-4: NMS + Gradient clipping
- Day 5: Testing and validation

**Week 2:**
- Day 1-3: Focal loss implementation
- Day 4-5: Mixed precision training

**Week 3:**
- Day 1-3: Faster R-CNN integration
- Day 4-5: Testing and comparison

**Week 4:**
- Day 1-3: YOLOv8 integration (optional)
- Day 4-5: Final testing and documentation

---

## Questions to Answer Before Starting

1. **Priority:** Accuracy or speed?
   - Accuracy → Faster R-CNN or YOLOv8
   - Speed → Keep EfficientDet, optimize inference

2. **Resources:** GPU memory available?
   - Limited → Smaller models, mixed precision
   - Plenty → Larger models, bigger batches

3. **Timeline:** How soon do you need results?
   - Urgent → Quick wins only (augmentation, scheduler, NMS)
   - Flexible → Full optimization pipeline

4. **Dependencies:** OK with external libraries?
   - Yes → YOLOv8 is great option
   - No → Stick with torchvision models

---

## Next Steps

1. Review this plan with team
2. Prioritize optimizations based on goals
3. Start with quick wins (Week 1)
4. Measure improvements after each change
5. Iterate based on results

For detailed explanations, see `MODEL_OPTIMIZATION_PLAN.md`.

