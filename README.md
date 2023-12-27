hi we are dhj


```
tinycd-template/
│
├── data/ - directory for storing input data
│   ├── LEVIR-CD 
│   │	├── train
│   │	├── val
│   │	├── test
│   │	└── list
│   ├── AERIAL-CD 
│   │	├── train
│   │	├── val
│   └── └── list
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
│   │	├── ESAMB.py
│   │	├── ffc_modules.py
│   └── └── TMM.py
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
└── requirements.txt - requirements 
```