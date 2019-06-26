from run import create_model
import argparse

model_path = 'model_mobilenetv2_v3_by_aimall_coat_upper_inner.h5'


parser = argparse.ArgumentParser()
args = parser.parse_args()

args.width =  224
args.base_model_name = 'mobilenetv2'

label_count = {'tag-upper-inner': 8, 'tag-coats-jackets': 3}

model = create_model(args, label_count)
print(model.summary())
model.load_weights(model_path)#'model_%s.h5' % 'mobilenetv2_v3_by_aimall_coat_upper_inner')
