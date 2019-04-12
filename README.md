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

* notebooks (driver code for running analysis) 
    * run_pipeline_comparisons.ipynb 
    * run_atlas_comparisons.ipynb
    * import_QC_data.ipynb
    * generate_plots.ipynb
    * learn_pipeline_transforms.ipynb
    
* lib (helper functions for data parsing and running analysis)
   * data_handling.py
   * data_stats.py
   * plot_utils.py
   * deeplearning.py
   
* scripts (code to extract useful data from pipeline output) 
    * get_dkt_data_civet.py 
    * get_vertex_data_fs.py
    * check_vertex_data.py


Legacy: 
- ./analysis, ./bin ! R scripts for data parsing, merging, and plotting (see https://github.com/companat/compare-surf-tools for details) 
