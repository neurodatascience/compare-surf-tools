# Project: compare_surf_tools
An extension of BrainHack Project (https://github.com/companat/compare-surf-tools) to compare thickness outputs from different pipelines run on ABIDE-I

## Objectives
- Compare output of preprocessing pipelines for structural MR imaging 
  - pipelines: Freesurfer (v5.1, v5.3, v6.0, ANTs, CIVET2.1)  
  - feature comparisons: ROI-wise (surface parcellations: DKT40, Destrieux, Glasser)
  - analytic comparisons: classifier performance (individual predictions), statistical inference (biological group differences)  
- Outlier detection 
  - identify outliers at differen scales 
  - identify outliers for different tasks 
    

## Data
Consolidated from the analysis results provided at http://preprocessed-connectomes-project.org/abide/Description, we provide the following unified data tables in the 'data' directory:
* ABIDE_Phenotype.csv             : phenotypic data for the subjects
* ABIDE_ants_thickness_data.csv   : thickness data from ANTS analysis
* ABIDE_fs5.3_LandRvolumes.csv    : volume data from FreeSurfer 5.3 analysis
* ABIDE_fs5.3_thickness.csv       : thickness data from FreeSurfer 5.3 analysis
* abide_fs5.1_landrvolumes.csv    : volume data from FreeSurfer 5.1 analysis
* cortical_fs5.1_measuresenigma_thickavg.csv : thickness data from FreeSurfer 5.1 analysis
* subject_check.csv               : summary table of the data available per subject

New dataset addition(s)
* Civet: data/ABIDE_civet2.1_thickness.csv (DKT) 
* FS: data/fs60_group_stats/* (DKT, Destrieux, Glasser) 


## Code
Current:     
* notebooks: driver code to run analysis 
* lib: helper functions for parsing and analysis
* scripts: code to read pipeline output (civet2.1, fs6.0) 
```
.
├──  notebooks           
│   ├── run_atlas_comparisons.ipynb
│   ├── run_pipeline_comparisons.ipynb
│   ├── import_QC_data.ipynb
│   ├── generate_plots.ipynb
│   └── learn_pipeline_transforms.ipynb
└── lib
│   ├── data_handling.py
│   ├── deeplearning.py
│   ├── data_stats.py
│   └── plot_utils.py
└── scripts
    ├── get_vertex_data_fs.py
    ├── get_dkt_data_civet.py
    └── check_vertex_data.py
```
Legacy: 
- see https://github.com/companat/compare-surf-tools for details

## Steps (see ./compute_workflow.png) 
Prereq: Processed output from a given pipeline (tool): e.g. FreeSurfer

A. Data parsing

1. run scripts/get_vertex_data_fs.py for each subject (ideally in a loop) to create summary CSV for all processed subjects. 
```
python get_vertex_data_fs.py -p $sub/surf -s '.fwhm20.fsaverage.mgh' -o ./fs_fsaverage_vout
```

2. run scripts/get_roi_data_fs.py on a FS subject dir to get ROI wise summay CSV for all subjects. Uses aparcstats2table command. 
```
python get_roi_data_fs.py -s ../data/subjects -l ../data/subject_list.txt -m thickness -p a2009s -o ../data/sample_output/
```
B. Data standardization 

C. Comparative analysis

D. Outlier detection
