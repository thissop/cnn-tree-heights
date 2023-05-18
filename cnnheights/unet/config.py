
# Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
# random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
# sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
patch_generation_stratergy = 'random' # 'random' or 'sequential'
patch_size = (256,256,4) # Height * Width * (Input + Output) channels

# Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
normalize = 1

# Shape of the input data, height*width*channel; Here channels are NVDI and Pan
input_shape = (256,256,2)
input_image_channel = [0,1]
input_label_channel = 2
input_weight_channel = 3

#
batch_size = 8
