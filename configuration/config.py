# Experiment
experiment_name              = 'cyclegan'

# Dataset
dataset_name                 = 'cardiac_segmentation'   # Dataset name
image_path                   = 'images'                 # relative path to image files. Should be subdirectories under this labeled for example 'A' and 'B'
image_path_subdirs           = ('A','B')                # subdirectories for real_image_A and real_image_B
#label_path                   = 'labels'                 # relative path to image files
trainA_file_path              = 'trainA.txt'              # relative path to training index file
trainB_file_path              = 'trainB.txt'              # relative path to training index file
testA_file_path               = 'testA.txt'               # relative path to testing index file
testB_file_path               = 'testB.txt'               # relative path to testing index file
data_suffix                  ='.png'
#label_suffix                 ='.png'
channel_size                 = 3
#classes                      = 10
target_size_train            = (320, 320)
target_size_test             = (320, 320)
target_size_valid            = (320, 320)
color_mode                   = 'grayscale'
#ignore_label                 = 10
#label_cval                   = 10
#cval                         = 0

loss_fn                      = 'softmax_sparse_crossentropy_ignoring_last_label'
metrics                      = 'sparse_accuracy_ignoring_last_label'
loss_shape                   = None

# Model
model_name                   = 'cyclegan' #

show_model                   = False           # Show the architecture layers
load_pretrained              = False           # Load a pretrained model for doing finetuning
weights_file                 = 'weights.hdf5'  # Training weight file name

# Run instructions
train_model                  = True            # Train the model
test_model                   = False           # Test the model
pred_model                   = False           # Predict using the model

# GAN parameters
lambda_1                     = 10.0            # Cyclic loss weight A_2_B
lambda_2                     = 10.0            # Cyclic loss weight B_2_A
lambda_D                     = 1.0             # Weight for loss from discriminator guess on synthetic images
generator_iterations         = 1               # Number of generator training iterations in each training loop
discriminator_iterations     = 1              # Number of generator training iterations in each training loop
synthetic_pool_size          = 50
use_identity_learning        = False
identity_mapping_modulus     = 10               # Identity mapping will be done each time the iteration number is divisable with this number
use_patchgan                 = True             # PatchGAN - if false the discriminator learning rate should be decreased
# Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
use_multiscale_discriminator = False
# Supervised learning part - for MR images - comparison
use_supervised_learning      = False
supervised_weight            = 10.0


# Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
use_resize_convolution       = False
learning_rate_base_D         = 2e-4
learning_rate_base_G         = 2e-4


# Debug
debug                        = False            # Use only few images for debuging
debug_images_train           = 50              # N images for training in debug mode (-1 means all)
debug_images_valid           = 30              # N images for validation in debug mode (-1 means all)
debug_images_test            = 30              # N images for testing in debug mode (-1 means all)
debug_n_epochs               = 2               # N of training epochs in debug mode

# Batch sizes
batch_size_train             = 1               # Batch size during training
batch_size_valid             = 1               # Batch size during validation
batch_size_test              = 1               # Batch size during testing\
batchnorm_momentum           = 0.95
batch_momentum               = 0.9

resume_training              = False

crop_mode                    = 'none'          # Crop or not
crop_size_train              = None            # Crop size during training (Height, Width) or None
crop_size_valid              = None            # Crop size during validation
crop_size_test               = None            # Crop size during testing

pad_size                     = None            # Pad size

# Data shuffle
shuffle_train                = True            # Whether to shuffle the training data
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data
seed_train                   = 1924            # Random seed for the training shuffle
seed_valid                   = 1924            # Random seed for the validation shuffle
seed_test                    = 1924            # Random seed for the testing shuffle

# Training parameters
optimizer                    = 'adam'           # Optimizer ['sgd' | 'adam' | 'nadam']
weight_decay                 = 0.00005         # Weight decay or L2 parameter norm penalty
n_epochs                     = 200            # Number of epochs during training
workers                      = 4

# Callback learning rate scheduler
LRScheduler_enabled          = False            # Enable the Callback
LRScheduler_batch_epoch      = 'epoch'          # Schedule the LR each 'batch' or 'epoch'
LRScheduler_type             = 'power_decay'    # Type of scheduler ['power_decay' | 'exp_decay' | 'adam' | 'progressive_drops']
LRScheduler_power            = 0.9              # Power for the power_decay, exp_decay modes

# Callback save results
save_results_enabled         = True            # Enable the Callback
save_results_nsamples        = 5               # Number of samples to save
save_results_batch_size      = 5               # Size of the batch
save_results_n_legend_rows   = 1               # Number of rows when showwing the legend

# Callback early stoping
earlyStopping_enabled        = True            # Enable the Callback
earlyStopping_monitor        = 'sparse_accuracy_ignoring_last_label'   # Metric to monitor
earlyStopping_mode           = 'max'           # Mode ['max' | 'min']
earlyStopping_patience       = 100             # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping

# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'sparse_accuracy_ignoring_last_label'   # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = True            # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 0               # Verbosity of the checkpoint

# Callback plot
plotHist_enabled             = True           # Enable the Callback
plotHist_verbose             = 0               # Verbosity of the callback

# Data augmentation for training and normalization
norm_fit_dataset                   = False   # If True it recompute std and mean from images. Either it uses the std and mean set at the dataset config file
norm_rescale                       = None # Scalar to divide and set range 0-1
norm_featurewise_center            = False   # Substract mean - dataset
norm_featurewise_std_normalization = False   # Divide std - dataset
norm_samplewise_center             = False  # Substract mean - sample
norm_samplewise_std_normalization  = False  # Divide std - sample
#cb_weights_method                  = 'median_freq_cost' # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']

# Data augmentation for training
da_rotation_range                  = 5      # Rnd rotation degrees 0-180
da_width_shift_range               = 0.0    # Rnd horizontal shift
da_height_shift_range              = 0.0    # Rnd vertical shift
da_shear_range                     = 0.0    # Shear in radians
da_zoom_range                      = [0.8, 1.2]    # Zoom
da_zoom_maintain_shape             = True   # keep same shape?
da_channel_shift_range             = 0.     # Channecf.l shifts
da_fill_mode                       = 'constant'  # Fill mode
da_horizontal_flip                 = False  # Rnd horizontal flip
da_vertical_flip                   = False  # Rnd vertical flip

da_save_to_dir                     = False  # Save the images for debuging
