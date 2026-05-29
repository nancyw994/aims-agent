# 📊 AIMS Agent Report Structure Guide

## New Report Structure Design Philosophy

**Design Principle: Logical Flow from Problem to Conclusion**

```
Why → What Data → Strategy → Execute → Results → Analysis → Next
(Problem) (Data) (Decision) (Action) (Outcome) (Meaning) (Future)
```

---

## 📋 Part 1: PROBLEM DEFINITION

### Purpose
Help readers quickly understand:
- What problem this project aims to solve
- Why this research is important
- What goals are expected to be achieved

### Content
1. **Motivation and Background**
   - Research motivation
   - Domain background knowledge
   
2. **Task Definition**
   - Dataset description
   - Target variable
   - Task type (regression/classification)
   - Clear research objectives

### Why This Order?
✅ Start with "why" to engage readers  
✅ Define clear goals to establish evaluation criteria

---

## 📊 Part 2: DATA UNDERSTANDING

### Purpose
Help readers comprehensively understand:
- Data quality
- Data distribution characteristics
- Relationships between features
- Potential modeling challenges

### Content
1. **Dataset Overview**
   - Number of samples, number of features
   - Data cleaning records

2. **Data Distribution Analysis**
   - Target variable distribution plot (Histogram)
   - Target vs key features relationship plots
   - Overall data distribution plot
   
3. **Feature Correlation Analysis**
   - Correlation heatmap
   
4. **LLM Dataset Analysis**
   - AI analysis of data quality
   - Modeling challenge identification
   
5. **Feature Selection Metrics**
   - Multicollinearity indicators
   - Nonlinearity estimation
   - Feature-to-sample ratio

### Why This Order?
✅ All "what the data looks like" content grouped together  
✅ From macro (distribution) to micro (correlation)  
✅ From visualization to quantitative metrics  
✅ Provides foundation for subsequent decisions

---

## 🎯 Part 3: STRATEGY & DECISION-MAKING

### Purpose
Help readers understand:
- **Why** these methods were chosen
- What the decision basis is
- What candidate options exist
- How to evaluate and select

### Content
1. **Feature Selection Strategy**
   - LLM recommended feature selection methods (with rationale)
   - CV evaluation results for each method
   - Final selected method + feature selection curve
   
2. **Model Selection Strategy**
   - LLM model family reasoning
   - Candidate model list
   - UQ capability analysis (not just accuracy)
   
3. **Hyperparameter Tuning Strategy**
   - LLM generated search spaces (with rationale)
   - Search strategy explanation

### Why This Order?
✅ **Clear cause-and-effect**: Data characteristics → Method selection → Rationale  
✅ **Transparent decisions**: Every choice backed by data and reasoning  
✅ **Select features first, then models, then tune parameters**: Follows actual workflow

---

## 🚀 Part 4: TRAINING & EXECUTION

### Purpose
Show readers:
- Whether training process was smooth
- Effects of parameter tuning
- Performance comparison of various models
- Final selected model

### Content
1. **Training Process Monitoring**
   - Loss vs Epochs (training convergence)
   - Accuracy/R² vs Epochs (performance improvement trajectory)
   
2. **Hyperparameter Tuning Results**
   - Hyperparameter tuning heatmap
   - CV stability heatmap
   
3. **Model Performance Comparison**
   - Comparison table of all candidate models
   - MSE/RMSE comparison charts
   
4. **Best Model Selection**
   - Best model name
   - Best hyperparameters
   - Why this is best

### Why This Order?
✅ **From process to results**: Training process → Tuning → Comparison → Selection  
✅ **Logical coherence**: See the process to understand results  
✅ **Support decisions**: Comparison data supports final choice

---

## 📈 Part 5: RESULTS ANALYSIS

### Purpose
Help readers deeply understand:
- Model prediction quality
- Error distribution
- Which features are most important
- Result reliability

### Content
1. **Prediction Quality Assessment**
   - Parity Plot (predicted vs actual)
   - Residual analysis plot
   
2. **Feature Importance Analysis**
   - Top important features table
   - Feature importance visualization

### Why This Order?
✅ **Overall first, then details**: First check prediction quality, then feature contribution  
✅ **Visualization + quantitative**: Charts + tables for dual verification  
✅ **Interpretability**: From black box to white box

---

## 💡 Part 6: CONCLUSIONS & RECOMMENDATIONS

### Purpose
Help readers clarify:
- Model reliability
- Key findings
- What to do next

### Content
1. **Model Reliability Assessment**
   - Conclusion strength rating
   - Factors affecting reliability
   - Explanation

2. **Key Findings Summary**
   - Best model
   - Best feature selection method
   - Top predictive features
   - Model quality assessment

3. **Recommendations for Next Steps**
   - Validation suggestions
   - Further research directions
   - Active learning recommendations
   - Production deployment considerations

### Why This Order?
✅ **Assess first, then summarize**: First discuss reliability, then list key findings  
✅ **Actionable**: Specific next-step recommendations  
✅ **Complete loop**: From problem to conclusion

---

## 📚 Part 7: APPENDIX

- Dataset Profile (detailed data profile)
- Generated Code (generated code)
- Self-Correction Logs (self-correction logs)

---

## 🎯 Before vs After Comparison

### ❌ Old Structure Problems

```
1. Motivation                    ← Good
2. Dataset Summary               ← Good
3. LLM Analysis                  ← OK position
4. Feature Selection Metrics     ← Too early, haven't seen data distribution
5. Feature Selection Methods     ← Decision process, but data analysis incomplete
6. Data Cleaning                 ← Should be in data understanding section
7. Data Distribution             ← TOO LATE! Should be earlier
8. Model Reasoning               ← Decision, OK position
9. Model Comparison              ← Results, OK position
10. Parity Plot                  ← Analysis, OK position
11. Feature Importance           ← Analysis, but separated from correlation heatmap
12. Correlation Heatmap          ← Should be in data understanding section
13. Hyperparameter Heatmap       ← Should be in training section
14. Feature Selection Curve      ← Should be in feature selection strategy section
15. Loss/Accuracy vs Epochs      ← Should be in training section
16. Why Best Model               ← Repetitive, should be in model selection section
17. Uncertainty Notes            ← Conclusion, OK position
```

**Problems:**
- Data distribution plots too late (#7), should be earlier
- Correlation heatmap and feature importance separated (#11 and #12)
- Training process plots (#15) and hyperparameter plots (#13) separated
- Feature selection curve (#14) and feature selection decisions (#5) separated
- Charts scattered, logic unclear

### ✅ New Structure Advantages

```
Part 1: Problem Definition
  - Motivation → Task Definition

Part 2: Data Understanding (all data-related content)
  - Overview → Distribution → Correlation → Analysis → Metrics

Part 3: Strategy & Decision-Making (all decision processes centralized)
  - Feature Selection (methods→evaluation→curve)
  - Model Selection (reasoning→candidates→UQ)
  - Hyperparameter Strategy

Part 4: Training & Execution (all training processes centralized)
  - Training Monitoring (Loss→Accuracy)
  - Hyperparameter Results (heatmap)
  - Model Comparison
  - Best Model

Part 5: Results Analysis (all analysis centralized)
  - Prediction Quality (Parity→Residuals)
  - Feature Importance

Part 6: Conclusions & Recommendations
  - Reliability → Summary → Recommendations
```

**Advantages:**
- ✅ Clear logic: Problem→Data→Decision→Execute→Analysis→Conclusion
- ✅ Related content grouped: Similar information together
- ✅ Clear cause-and-effect: Every decision has prerequisite foundation
- ✅ Easy to understand: Follows human thought process
- ✅ Reproducible: Clearly shows entire workflow

---

## 🎓 Usage Recommendations

### For Readers
1. **Quick browse**: Look at Part titles to understand overall flow
2. **Deep reading**: Read in order for logical coherence
3. **Jump reading**: Only interested in certain parts (e.g., results analysis), can jump directly

### For Developers
1. When **generating reports**, organize content according to new structure
2. When **adding new features**, consider which Part they belong to
3. When **debugging issues**, easy to locate corresponding section

---

## 📝 Summary

**Core improvement:** From "listing content" to "telling a story"

Old report like "information accumulation", new report like a "research paper":
- Clear beginning, development, turn, and conclusion
- Each section has clear purpose
- Logical flow, clear cause-and-effect
- Readers can easily understand "why do it this way" and "what the results mean"

---

## Complete Structure Template

```
PART 1: PROBLEM DEFINITION
├── 1.1 Motivation and Background
└── 1.2 Task Definition

PART 2: DATA UNDERSTANDING
├── 2.1 Dataset Overview
├── 2.2 Data Distribution Analysis
├── 2.3 Feature Correlation Analysis
├── 2.4 LLM Dataset Analysis
└── 2.5 Feature Selection Metrics

PART 3: STRATEGY & DECISION-MAKING
├── 3.1 Feature Selection Strategy
│   ├── 3.1.1 LLM Recommendations
│   ├── 3.1.2 Method Evaluation
│   └── 3.1.3 Selection Curve
├── 3.2 Model Selection Strategy
│   ├── 3.2.1 LLM Model-Family Reasoning
│   └── 3.2.2 UQ Capability Analysis
└── 3.3 Hyperparameter Tuning Strategy
    └── 3.3.1 LLM Search Spaces

PART 4: TRAINING & EXECUTION
├── 4.1 Training Process Monitoring
│   ├── 4.1.1 Loss Convergence
│   └── 4.1.2 Accuracy/R² Evolution
├── 4.2 Hyperparameter Tuning Results
├── 4.3 Model Performance Comparison
└── 4.4 Best Model Selection
    ├── 4.4.1 Optimal Hyperparameters
    ├── 4.4.2 Why These Hyperparameters Are Best
    └── 4.4.3 Why This Model Is Best Overall

PART 5: RESULTS ANALYSIS
├── 5.1 Prediction Quality Assessment
│   ├── 5.1.1 Parity Plot
│   └── 5.1.2 Residual Analysis
└── 5.2 Feature Importance Analysis
    ├── 5.2.1 Top Important Features
    └── 5.2.2 Feature Importance Visualization

PART 6: CONCLUSIONS & RECOMMENDATIONS
├── 6.1 Model Reliability Assessment
├── 6.2 Key Findings Summary
└── 6.3 Recommendations for Next Steps

APPENDIX
├── Dataset Profile Details
├── Generated Code
└── Self-Correction Logs
```

---

## Benefits Summary

### 1. Clear Logical Flow
**Old**: Jump around, hard to follow  
**New**: Linear progression, easy to understand

### 2. Grouped Related Content
**Old**: Related charts scattered  
**New**: All related information together

### 3. Explicit Cause-and-Effect
**Old**: Decisions before seeing data  
**New**: Data → Analysis → Decision → Action

### 4. Professional Presentation
**Old**: Information dump  
**New**: Research narrative with structure

### 5. Reproducible Workflow
**Old**: Unclear how decisions were made  
**New**: Every step documented with rationale

---

## Implementation Status

✅ Report structure implemented in `model_strategy_analysis.py`  
✅ All parts use clean English headers  
✅ Logical flow preserved  
✅ All visualizations and tables included  
✅ Documentation complete

**Next Steps:**
- Run pipeline to generate sample report
- Review and gather feedback
- Iterate on improvements
