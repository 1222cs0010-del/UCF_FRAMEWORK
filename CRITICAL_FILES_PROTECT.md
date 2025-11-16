# 🔒 CRITICAL FILES - DO NOT DELETE

**Status**: Protected  
**Date**: November 16, 2025  
**Importance**: CRITICAL - Core Framework

---

## 🛡️ Core Pipeline Files (MUST KEEP)

### 1. `ucf_pipeline_final.py` ⭐ CRITICAL

**Purpose**: Main unified pipeline orchestrator  
**Lines**: 1304  
**Status**: ✅ PROTECTED - DO NOT DELETE

**Content**:
- Complete 5-stage pipeline implementation
- Stage 1: Load & Preprocess (0.01s)
- Stage 2: CA-KD Distillation (25-35s)
- Stage 3: CSM Streaming (0.07s)
- Stage 4: Quantization (0.20s)
- Stage 5: Fairness Evaluation (0.87s)

**Dependencies**:
- Imports from `ucf_data_utils.py`
- Uses 6 integrated baselines
- Coordinates all pipeline stages

**Why Keep**: 
- This is the MAIN FRAMEWORK
- Contains 5-stage pipeline orchestration
- All other tools depend on this
- Months of development invested

---

### 2. `ucf_data_utils.py` ⭐ CRITICAL

**Purpose**: Data loading and preprocessing utilities  
**Status**: ✅ PROTECTED - DO NOT DELETE

**Content**:
- UCFDataLoader class
- Dataset loading (SVAMP, GSM8K, ShareGPT, etc.)
- Data caching and optimization
- Tokenization utilities

**Why Keep**:
- Required by `ucf_pipeline_final.py`
- Handles all dataset operations
- Critical for data pipeline

---

### 3. `ucf_core/` Directory ⭐ CRITICAL

**Purpose**: Core pipeline implementations  
**Status**: ✅ PROTECTED - DO NOT DELETE

**Content**:
- Core algorithm implementations
- Baseline integration code
- Pipeline stage implementations

**Why Keep**:
- Contains actual algorithm implementations
- Required by pipeline

---

## 📦 Supporting Files (KEEP)

### Configuration Files
- `configs/default.yaml` - Default configuration ✓
- `configs/unified_experiment.yaml` - Experiment config ✓
- `requirements.txt` - Python dependencies ✓
- `environment.yml` - Conda environment ✓

### Integration Files
- `baselines/` - All baseline implementations ✓
- `tests/` - Test suite ✓
- `utils/` - Utility functions ✓
- `tools/` - Helper tools ✓

---

## 🆕 New Training Files (KEEP)

Recently added for 1-hour training demonstration:

- `one_hour_training.py` ✓ New optimized training script
- `expected_results.py` ✓ Expected metrics and benchmarks
- `generate_demo_report.py` ✓ Report generation
- `ONE_HOUR_QUICK_START.md` ✓ Quick start guide
- `ONE_HOUR_EXECUTION_GUIDE.md` ✓ Execution guide

---

## 🗑️ Safe to Delete (Optional Cleanup)

These files are documentation/log files that can be safely deleted if needed:

- `GITHUB_PUSH_COMPLETE.md` - Summary document
- `CODE_PATH_VISUALIZATION.txt` - Analysis document
- `DISTILLATION_CODE_ANALYSIS.md` - Analysis document
- `STEP2_0_00s_COMPLETE_ANSWER.md` - Analysis document
- `*.md` - Most analysis markdown files (keep START_HERE.md, README.md)

---

## 📋 File Hierarchy

```
/mnt/d/ucf_framework/
├── 🔒 CRITICAL CORE FILES
│   ├── ucf_pipeline_final.py          ⭐ MAIN PIPELINE
│   ├── ucf_data_utils.py              ⭐ DATA UTILITIES
│   ├── ucf_core/                      ⭐ CORE IMPLEMENTATIONS
│   ├── baselines/                     ⭐ BASELINE IMPLEMENTATIONS
│   ├── configs/                       ⭐ CONFIGURATIONS
│   └── requirements.txt               ⭐ DEPENDENCIES
│
├── 📚 NEW 1-HOUR TRAINING FILES
│   ├── one_hour_training.py           ✓ Keep
│   ├── expected_results.py            ✓ Keep
│   ├── generate_demo_report.py        ✓ Keep
│   ├── ONE_HOUR_QUICK_START.md        ✓ Keep
│   └── ONE_HOUR_EXECUTION_GUIDE.md    ✓ Keep
│
├── 📖 DOCUMENTATION (KEEP IMPORTANT ONES)
│   ├── README.md                      ✓ Keep
│   ├── START_HERE.md                  ✓ Keep
│   ├── DEPLOYMENT_GUIDE.md            ✓ Keep
│   └── ... other analysis docs        Optional
│
└── 🗂️ SUPPORT FILES
    ├── tests/                         ✓ Keep
    ├── scripts/                       ✓ Keep
    ├── utils/                         ✓ Keep
    └── tools/                         ✓ Keep
```

---

## ⚠️ PROTECTION RULES

### DO NOT DELETE:
1. ❌ `ucf_pipeline_final.py` - MAIN FRAMEWORK
2. ❌ `ucf_data_utils.py` - DATA LOADING
3. ❌ `ucf_core/` - CORE IMPLEMENTATIONS
4. ❌ `baselines/` - BASELINE CODE
5. ❌ `configs/` - CONFIGURATION
6. ❌ `requirements.txt` - DEPENDENCIES
7. ❌ `one_hour_training.py` - NEW TRAINING SCRIPT
8. ❌ `README.md` - PROJECT README

### OK TO DELETE (Optional):
- ✓ Analysis documents (DISTILLATION_CODE_ANALYSIS.md, etc.)
- ✓ Summary files (GITHUB_PUSH_COMPLETE.md, etc.)
- ✓ Temporary logs

### MUST KEEP:
- ✓ All Python source files (.py)
- ✓ All configuration files (.yaml, .yml, .txt)
- ✓ Core documentation (README.md, START_HERE.md)
- ✓ Baseline implementations in baselines/

---

## 🔧 Git Status

All critical files are:
- ✅ Tracked in git
- ✅ Committed to GitHub
- ✅ Backed up in remote repository
- ✅ Safe to keep locally

**Repository**: https://github.com/1222cs0010-del/UCF_FRAMEWORK

---

## 📌 Summary

**ucf_pipeline_final.py** is the HEART of the framework:
- Contains the complete 5-stage pipeline
- 1,304 lines of carefully designed code
- 5+ stages of processing
- 6 integrated baselines
- Production-ready implementation

**Never delete this file!**

---

**Last Updated**: November 16, 2025  
**Status**: ✅ All Critical Files Protected
