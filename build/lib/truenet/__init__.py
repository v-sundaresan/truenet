'''
Triplanar U-Net ensemble network (TrUE-Net)

For training, run:
truenet train -i <input_directory> -m <model_directory>

For testing, run:
truenet evaluate -i <input_directory> -m <model_directory> -o <output_directory>

For leave-one-out validation, run:
truenet loo_validate -i <input_directory> -o <output_directory>

for fine-tuning, run:
truenet fine_tune -i <input_directory> -m <model_directory> -o <output_directory>
'''