# YOLOv5 Hyperparameter Tuning Plan

## Executive Summary

This document outlines comprehensive hyperparameter tuning strategies for the YOLOv5 model that **do not modify the model architecture**. All optimizations focus on training dynamics, loss functions, optimizers, and learning rate schedules to improve model performance on the SimSurgSkill dataset.

## Current State Analysis

### Current Configuration
- **Optimizer**: Adam (fixed learning rate: 0.001)
- **Learning Rate**: Fixed at 0.001 (no scheduling)
- **Loss Weights**: coord=0.05, conf=1.0, class=0.5
- **Gradient Clipping**: max_norm=10.0
- **Weight Decay**: None (0.0)
- **Batch Normalization**: Default PyTorch settings (momentum=0.1, eps=1e-5)
- **Input Size**: 640x640
- **Model Size**: Small (s) by default
- **Epochs**: 10 (default, likely insufficient)
- **Batch Size**: 4 (default)

### Identified Gaps
1. ❌ No learning rate scheduling (despite documentation mentioning it)
2. ❌ No differential learning rates for backbone vs detection heads
3. ❌ No weight decay regularization
4. ❌ Suboptimal loss weights (may need tuning for surgical instruments)
5. ❌ Fixed learning rate may be too high/low for convergence
6. ❌ Limited epochs (10 is typically insufficient for object detection)
7. ❌ Batch normalization parameters not optimized

---

## Option 1: Conservative Incremental Approach
**Best for**: Stable training, gradual improvements, avoiding instability

### Philosophy
Make small, proven improvements one at a time, ensuring each change improves performance before adding the next.

### Hyperparameter Changes

#### 1.1 Learning Rate Scheduling
- **Warmup**: Linear warmup for first 3 epochs (10% of 30 epochs)
- **Scheduler**: Cosine Annealing LR
  - Initial LR: 0.001
  - Minimum LR: 1e-5
  - T_max: 27 epochs (after warmup)
- **Rationale**: Prevents early training instability, ensures smooth convergence

#### 1.2 Optimizer Enhancement
- **Switch to**: AdamW (Adam with decoupled weight decay)
- **Weight Decay**: 0.0005
- **Betas**: (0.937, 0.999) - YOLOv5 standard
- **Epsilon**: 1e-8
- **Rationale**: Better generalization, decoupled weight decay prevents overfitting

#### 1.3 Loss Weight Tuning
- **Coordinate Loss**: 0.05 → **0.1** (better localization for small surgical instruments)
- **Confidence Loss**: 1.0 → **1.5** (reduce false negatives)
- **Classification Loss**: 0.5 → **0.8** (improve class discrimination)
- **Rationale**: Surgical instruments are small objects, need stronger localization signal

#### 1.4 Training Duration
- **Epochs**: 10 → **30-50**
- **Early Stopping**: Patience=10 epochs (stop if validation IoU doesn't improve)
- **Rationale**: Object detection models need more epochs to converge

#### 1.5 Batch Normalization Tuning
- **Momentum**: 0.1 → **0.03** (faster adaptation to batch statistics)
- **Epsilon**: 1e-5 → **1e-4** (more stable for small batches)
- **Rationale**: Better normalization for small batch sizes

### Expected Improvements
- **IoU**: +3-5%
- **mAP**: +5-8%
- **Training Stability**: Significantly improved
- **Convergence Speed**: Faster convergence

### Implementation Complexity
- **Difficulty**: ⭐⭐ (Easy-Medium)
- **Risk**: Low (proven techniques)
- **Time**: 2-3 hours implementation + 1-2 days training

---

## Option 2: Aggressive Optimization Approach
**Best for**: Maximum performance, willing to experiment, have computational resources

### Philosophy
Apply all proven optimization techniques simultaneously, including advanced strategies.

### Hyperparameter Changes

#### 2.1 Advanced Learning Rate Scheduling
- **Warmup**: Linear warmup for first 5 epochs
- **Scheduler**: Cosine Annealing with Restarts (warm restarts every 20 epochs)
- **Initial LR**: 0.001
- **Minimum LR**: 5e-6
- **Restart Multiplier**: 0.8 (each restart starts at 80% of previous max)
- **Rationale**: Escapes local minima, better exploration of loss landscape

#### 2.2 Differential Learning Rates
- **Backbone (ResNet)**: 0.0001 (10x lower than base)
- **Detection Heads**: 0.001 (full rate)
- **PANet Layers**: 0.0005 (5x lower, intermediate)
- **Rationale**: Pretrained backbone needs fine-tuning, heads need full learning

#### 2.3 Optimizer: SGD with Momentum (Alternative)
- **Optimizer**: SGD with Nesterov momentum
- **Learning Rate**: 0.01 (10x higher than Adam)
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Rationale**: SGD often achieves better final performance than Adam for object detection

#### 2.4 Advanced Loss Weight Tuning
- **Coordinate Loss**: 0.1 (with IoU loss component)
- **Confidence Loss**: 2.0 (with focal loss option)
- **Classification Loss**: 1.0
- **Focal Loss Gamma**: 2.0 (if using focal loss variant)
- **Label Smoothing**: 0.1 (optional)
- **Rationale**: Focal loss helps with class imbalance, IoU loss improves localization

#### 2.5 Multi-Scale Training
- **Input Sizes**: [512, 640, 768] (randomly select per epoch)
- **Rationale**: Improves robustness to different object scales

#### 2.6 Gradient Clipping Enhancement
- **Max Norm**: 10.0 → **5.0** (tighter clipping)
- **Clip Type**: Global norm (current) + per-parameter clipping for outliers
- **Rationale**: Prevents gradient explosions more effectively

#### 2.7 Extended Training
- **Epochs**: 50-100
- **Learning Rate Drops**: Reduce LR by 0.1x at epochs 40, 60, 80
- **Rationale**: Extended training with careful LR management

### Expected Improvements
- **IoU**: +8-12%
- **mAP**: +10-15%
- **Robustness**: Significantly improved
- **Final Performance**: State-of-the-art for this dataset

### Implementation Complexity
- **Difficulty**: ⭐⭐⭐⭐ (Hard)
- **Risk**: Medium (some techniques may conflict)
- **Time**: 4-6 hours implementation + 3-5 days training

---

## Option 3: Balanced Production-Ready Approach
**Best for**: Best performance/effort ratio, production deployment, reliable results

### Philosophy
Apply well-tested optimizations that provide maximum benefit with minimal risk. This is the recommended approach.

### Hyperparameter Changes

#### 3.1 Learning Rate Schedule (Cosine with Warmup)
- **Warmup Epochs**: 5 (10% of 50 epochs)
- **Warmup Type**: Linear warmup from 0.0001 to 0.001
- **Scheduler**: Cosine Annealing LR
  - Initial LR: 0.001
  - Minimum LR: 1e-5
  - T_max: 45 epochs
- **Rationale**: Proven schedule, prevents early instability

#### 3.2 Optimizer: AdamW
- **Learning Rate**: 0.001 (base)
- **Weight Decay**: 0.0005
- **Betas**: (0.9, 0.999) - standard AdamW
- **Epsilon**: 1e-8
- **Rationale**: Better than Adam, well-tested

#### 3.3 Differential Learning Rates
- **Backbone Parameters**: 0.0001 (0.1x base LR)
- **Detection Heads**: 0.001 (1.0x base LR)
- **PANet Fusion Layers**: 0.0005 (0.5x base LR)
- **Rationale**: Fine-tune pretrained backbone, train heads from scratch

#### 3.4 Optimized Loss Weights
- **Coordinate Loss**: 0.1 (2x increase for better localization)
- **Confidence Loss**: 1.5 (1.5x increase for better detection)
- **Classification Loss**: 0.8 (1.6x increase for better discrimination)
- **Rationale**: Balanced improvement across all loss components

#### 3.5 Batch Normalization Optimization
- **Momentum**: 0.03 (faster adaptation)
- **Epsilon**: 1e-4 (stability for small batches)
- **Track Running Stats**: True (always)
- **Rationale**: Better for small batch training

#### 3.6 Training Configuration
- **Epochs**: 50 (sufficient for convergence)
- **Early Stopping**: 
  - Metric: Validation IoU
  - Patience: 10 epochs
  - Min Delta: 0.001
- **Gradient Clipping**: 10.0 (keep current)
- **Rationale**: Sufficient training with automatic stopping

#### 3.7 Learning Rate Finder (Optional)
- **Method**: Run LR range test before training
- **Range**: 1e-6 to 1e-1
- **Rationale**: Find optimal initial learning rate for this dataset

### Expected Improvements
- **IoU**: +5-8%
- **mAP**: +8-12%
- **Training Stability**: Excellent
- **Convergence**: Reliable and predictable

### Implementation Complexity
- **Difficulty**: ⭐⭐⭐ (Medium)
- **Risk**: Low-Medium (well-tested techniques)
- **Time**: 3-4 hours implementation + 2-3 days training

---

## Option 4: Research-Oriented Experimental Approach
**Best for**: Maximum learning, experimentation, research purposes

### Philosophy
Try cutting-edge techniques and experimental optimizations to push performance boundaries.

### Hyperparameter Changes

#### 4.1 Advanced Learning Rate Schedules
- **Option A**: OneCycleLR (fast convergence)
  - Max LR: 0.01
  - Min LR: 1e-5
  - Pct Start: 0.3 (30% of training at max LR)
- **Option B**: Polynomial Decay with Warmup
  - Power: 0.9
  - Warmup: 5 epochs
- **Option C**: ReduceLROnPlateau (adaptive)
  - Factor: 0.5
  - Patience: 5 epochs
  - Min LR: 1e-6
- **Rationale**: Test which schedule works best for this dataset

#### 4.2 Optimizer Comparison
- **Test Multiple Optimizers**:
  1. AdamW (baseline)
  2. SGD with Nesterov (momentum=0.937)
  3. Adam with Lookahead wrapper
  4. RAdam (Rectified Adam)
- **Rationale**: Find optimal optimizer for this specific task

#### 4.3 Advanced Loss Functions
- **GIoU Loss**: Replace MSE for coordinate loss
- **Focal Loss**: For confidence and classification
- **Label Smoothing**: 0.1 for classification
- **Loss Weight Scheduling**: Increase coord weight over time
- **Rationale**: Better loss functions improve learning

#### 4.4 Hyperparameter Search
- **Method**: Bayesian Optimization (Optuna)
- **Search Space**:
  - Learning rate: [1e-4, 1e-2]
  - Weight decay: [1e-5, 1e-3]
  - Loss weights: coord [0.05, 0.2], conf [1.0, 2.0], class [0.5, 1.0]
  - Batch size: [2, 8]
  - Gradient clip: [5.0, 15.0]
- **Trials**: 50-100
- **Rationale**: Systematic search for optimal hyperparameters

#### 4.5 Advanced Training Techniques
- **Mixup Augmentation**: Alpha=0.2
- **CutMix Augmentation**: Alpha=1.0
- **Mosaic Augmentation**: 4-image mosaic
- **Self-Supervised Pretraining**: Contrastive learning on unlabeled data
- **Rationale**: Data augmentation improves generalization

#### 4.6 Ensemble Learning
- **Multiple Models**: Train 3-5 models with different seeds
- **Averaging**: Weighted average of predictions
- **Rationale**: Ensemble improves final performance

### Expected Improvements
- **IoU**: +10-15% (with best configuration)
- **mAP**: +12-18%
- **Robustness**: Significantly improved
- **Research Value**: High (publishable results)

### Implementation Complexity
- **Difficulty**: ⭐⭐⭐⭐⭐ (Very Hard)
- **Risk**: High (experimental techniques)
- **Time**: 1-2 weeks implementation + 1-2 weeks training/search

---

## Detailed Implementation Plan (Option 3 - Recommended)

### Phase 1: Learning Rate Scheduling (Priority: High)
**Files to Modify**: `main.py`

1. **Add Learning Rate Scheduler**:
   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
   
   # Warmup scheduler
   warmup_scheduler = LinearLR(
       optimizer, 
       start_factor=0.1,  # Start at 10% of LR
       end_factor=1.0,
       total_iters=5  # 5 epochs warmup
   )
   
   # Cosine annealing scheduler
   cosine_scheduler = CosineAnnealingLR(
       optimizer,
       T_max=45,  # 45 epochs after warmup
       eta_min=1e-5
   )
   
   # Combined scheduler
   scheduler = SequentialLR(
       optimizer,
       schedulers=[warmup_scheduler, cosine_scheduler],
       milestones=[5]
   )
   ```

2. **Update Training Loop**:
   - Call `scheduler.step()` after each epoch
   - Log learning rate to W&B

**Expected Impact**: +2-3% IoU improvement

---

### Phase 2: Differential Learning Rates (Priority: High)
**Files to Modify**: `main.py` (setup_device_and_model function)

1. **Separate Model Parameters**:
   ```python
   # Identify backbone parameters
   backbone_params = []
   head_params = []
   panet_params = []
   
   for name, param in model.named_parameters():
       if 'backbone' in name:
           backbone_params.append(param)
       elif 'detect_scale' in name or 'fusion' in name:
           head_params.append(param)
       else:
           panet_params.append(param)
   
   # Create optimizer with different LRs
   optimizer = optim.AdamW([
       {'params': backbone_params, 'lr': learning_rate * 0.1},
       {'params': panet_params, 'lr': learning_rate * 0.5},
       {'params': head_params, 'lr': learning_rate}
   ], lr=learning_rate, weight_decay=0.0005)
   ```

**Expected Impact**: +1-2% IoU improvement

---

### Phase 3: Loss Weight Optimization (Priority: Medium)
**Files to Modify**: `src/models/yolov5.py` (YOLOv5Model.__init__)

1. **Update Default Loss Weights**:
   ```python
   if loss_weights is None:
       self.loss_weights = {
           'coord': 0.1,    # Increased from 0.05
           'conf': 1.5,      # Increased from 1.0
           'class': 0.8      # Increased from 0.5
       }
   ```

2. **Make Configurable via Variants**:
   - Add to `model_config.py` variants
   - Allow override via environment variables

**Expected Impact**: +1-2% IoU improvement

---

### Phase 4: Batch Normalization Tuning (Priority: Low)
**Files to Modify**: `src/models/yolov5.py` (_make_yolov5_head)

1. **Update BatchNorm Parameters**:
   ```python
   nn.BatchNorm2d(in_channels, momentum=0.03, eps=1e-4)
   ```

**Expected Impact**: +0.5-1% IoU improvement, better training stability

---

### Phase 5: Extended Training (Priority: High)
**Files to Modify**: `main.py`, `env.example`

1. **Update Default Epochs**:
   - Change default from 10 to 50
   - Add early stopping logic

2. **Early Stopping Implementation**:
   ```python
   best_val_iou = 0.0
   patience = 10
   patience_counter = 0
   
   # In training loop:
   if val_metrics['mean_iou'] > best_val_iou:
       best_val_iou = val_metrics['mean_iou']
       patience_counter = 0
   else:
       patience_counter += 1
       if patience_counter >= patience:
           logger.info("Early stopping triggered")
           break
   ```

**Expected Impact**: +2-4% IoU improvement (more training time)

---

## Hyperparameter Search Strategy

### Manual Grid Search (Recommended First Step)
1. **Learning Rate**: [0.0005, 0.001, 0.002]
2. **Weight Decay**: [0.0001, 0.0005, 0.001]
3. **Loss Weights**: 
   - coord: [0.05, 0.1, 0.15]
   - conf: [1.0, 1.5, 2.0]
   - class: [0.5, 0.8, 1.0]

**Total Combinations**: 3 × 3 × 3 × 3 × 3 = 243 (can be reduced with prior knowledge)

### Bayesian Optimization (Advanced)
- Use Optuna or Ray Tune
- Search space defined above
- 50-100 trials
- Optimize for validation IoU

---

## Monitoring and Evaluation

### Key Metrics to Track
1. **Training Loss**: Should decrease smoothly
2. **Validation Loss**: Should track training loss
3. **Learning Rate**: Should follow schedule
4. **Gradient Norm**: Should stay below clipping threshold
5. **Validation IoU**: Primary metric for model selection
6. **Validation mAP**: Secondary metric
7. **Precision/Recall**: Class-specific metrics

### W&B Logging Enhancements
- Log learning rate schedule
- Log loss components separately (coord, conf, class)
- Log gradient norms per parameter group
- Log validation metrics every epoch
- Create custom plots for hyperparameter analysis

---

## Risk Assessment

### Low Risk Changes
- ✅ Learning rate scheduling
- ✅ Weight decay addition
- ✅ Extended training epochs
- ✅ Loss weight tuning

### Medium Risk Changes
- ⚠️ Differential learning rates (may need careful tuning)
- ⚠️ Batch normalization changes (may affect convergence)
- ⚠️ Optimizer switching (may need LR adjustment)

### High Risk Changes
- ⚠️ Advanced loss functions (may break training)
- ⚠️ Multi-scale training (may cause memory issues)
- ⚠️ Experimental optimizers (unproven)

---

## Recommended Implementation Order

### Week 1: Foundation
1. **Day 1-2**: Implement learning rate scheduling (Option 3, Phase 1)
2. **Day 3-4**: Add differential learning rates (Option 3, Phase 2)
3. **Day 5**: Update loss weights (Option 3, Phase 3)
4. **Day 6-7**: Run initial training (30 epochs), evaluate results

### Week 2: Optimization
1. **Day 1-2**: Fine-tune hyperparameters based on Week 1 results
2. **Day 3-4**: Implement early stopping
3. **Day 5-7**: Run extended training (50 epochs), final evaluation

### Week 3: Advanced (Optional)
1. **Day 1-3**: Batch normalization tuning
2. **Day 4-5**: Hyperparameter search (if needed)
3. **Day 6-7**: Final model training and evaluation

---

## Success Criteria

### Minimum Success
- IoU improvement: +3%
- mAP improvement: +5%
- Training stability: No crashes, smooth loss curves

### Target Success
- IoU improvement: +5-8%
- mAP improvement: +8-12%
- Training stability: Excellent, predictable convergence

### Stretch Goal
- IoU improvement: +10%+
- mAP improvement: +15%+
- State-of-the-art performance for SimSurgSkill dataset

---

## Conclusion

**Recommended Approach**: **Option 3 (Balanced Production-Ready)**

This approach provides the best balance of:
- Performance improvement (expected +5-8% IoU)
- Implementation effort (3-4 hours)
- Risk level (low-medium)
- Training time (2-3 days)

The plan focuses on proven techniques that have shown consistent improvements across object detection tasks while avoiding experimental methods that may introduce instability.

**Next Steps**:
1. Review and approve this plan
2. Select preferred option (or combination)
3. Begin implementation starting with Phase 1
4. Monitor results and iterate

---

## Appendix: Hyperparameter Reference Table

| Hyperparameter | Current | Option 1 | Option 3 (Recommended) | Option 2 |
|---------------|---------|----------|------------------------|----------|
| **Optimizer** | Adam | AdamW | AdamW | SGD+Nesterov |
| **Base LR** | 0.001 | 0.001 | 0.001 | 0.01 |
| **Backbone LR** | 0.001 | 0.0001 | 0.0001 | 0.001 |
| **Head LR** | 0.001 | 0.001 | 0.001 | 0.01 |
| **Weight Decay** | 0.0 | 0.0005 | 0.0005 | 0.0005 |
| **LR Schedule** | None | Cosine+Warmup | Cosine+Warmup | Cosine+Restarts |
| **Warmup Epochs** | 0 | 3 | 5 | 5 |
| **Epochs** | 10 | 30-50 | 50 | 50-100 |
| **Coord Loss Weight** | 0.05 | 0.1 | 0.1 | 0.1 |
| **Conf Loss Weight** | 1.0 | 1.5 | 1.5 | 2.0 |
| **Class Loss Weight** | 0.5 | 0.8 | 0.8 | 1.0 |
| **BN Momentum** | 0.1 | 0.03 | 0.03 | 0.03 |
| **BN Epsilon** | 1e-5 | 1e-4 | 1e-4 | 1e-4 |
| **Grad Clip** | 10.0 | 10.0 | 10.0 | 5.0 |
| **Early Stopping** | No | Yes | Yes | Yes |

---

*Document Version: 1.0*  
*Last Updated: 2024*  
*Author: AI Assistant*

