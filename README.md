hi we are dhj

```
tinycd-template/
│
├── data/ - directory for storing input data
│   ├── LEVIR-CD - LEVIR dataset
│   │	├── train
│   │	├── val
│   │	├── test
│   │	└── list
│   ├── AERIAL-CD - AI-hub custom dataset
│   │   └── ...
│   ├── ONERA-CD - Onera dataset
│   │   └── ...
│   ├── S2looking - S2looking dataset
│   │   └── ...
│   ├── WHU-CD - WHU building dataset
│   │   └── ...
│   └── INFERENCE-CD - dataset for Inference
│       └── ...
│
├── dataset/ - anything about datasets goes here 
│   ├── dataset.py 
│   └── inference_dataset.py 
│
├── metrics/ - metric calculator 
│   └── metric_tool.py
│
├── models/ - directory for change detection model
│   └── tinycd.py 
│
├── modules/ - modules for change detection model
│   ├── ESAMM.py
│   ├── ffc_modules.py
│   └── SMM.py
│
├── outputs/ - directory for saving train results
│   ├── best_weights/ - directory for saving best model weights
│   ├── inference_output/ - directory for saving inference outputs
│   └── train_result/ - directory for saving trained model weights
│
├── utils/ - small utility functions
│   └── utils.py
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
├── inference.py - inference of trained model
│
├── READMD.md - readme! 
├── txt_generator.py - list txt file generator for LEVIR-CD structure dataset 
└── requirements.txt - requirements for PIP
```