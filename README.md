# Project: compare_surf_tools
An extension of BrainHack Project (https://github.com/companat/compare-surf-tools) to compare thickness outputs from different pipelines run on ABIDE-I

## Objectives
- Compare output of preprocessing pipelines for structural MR imaging 
  - pipelines: Freesurfer, ANTs, CIVET 
  - feature comparisons: vertex-wise, ROI-wise (atlas)
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
* aparc_lh_thickness_table.txt : left hemisphere thickness data from FreeSurfer 6.0 analysis
* aparc_rh_thickness_table.txt : right hemisphere thickness data from FreeSurfer 6.0 analysis


## Code
Current: 
- ./lib : python functions for data parsing and running statistical analysis 
- ./notebooks/run_pipeline_comparisons.ipynb : driver notebook for running analysis

Legacy: 
- ./analysis, ./bin ! R scripts for data parsing, merging, and plotting (see https://github.com/companat/compare-surf-tools for details) 
