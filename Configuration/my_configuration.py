
# running mode: allowed values: terminal or remote
# terminal --> this expects to run your code from local computer using terminal and you should specify the
#  DATA_DIR, OUTPUT_DIR, and NUM_CPU arguments in this Configuration file (SEE BELOW)

# remote --> this running the code in remote high performance computer, you need to specify DATA_DIR,
# OUTPUT_DIR, and NUM_CPU arguments your remote job submission file  (SEE BELOW)
running_mode = 'terminal'


# -------------------- THESE PARAMETERS WILL BE USED ONLY WHEN running_mod='terminal' -----------------------
num_cpu=12
data_dir="/path/to/the/code/directory" # /data/AA/BB/data
output_dir="/data/AA/BB/output"
# -----------------------------------------------------------------------------------------------------------

# cell names
cell_names_ordered = ('CD4+FOXP-', 'CD8+', 'FOXP3+CD4+')

# THESE ARE OPTIMIZED VALUES for cell detection and classification on validdation data
pred_prob_threshold = 0.5	# cell detection probability threshold
area_threshold = 3	# binary objects with area smaller than 3 pixels^2 are treated as nosy and removed