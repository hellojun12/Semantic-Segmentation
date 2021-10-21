import wandb

def _labels(category_names):
  l = {}
  for i, label in enumerate(category_names):
    l[i] = label
  return l

def wb_mask(bg_img, pred_mask, true_mask, category_names):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : _labels(category_names)},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : _labels(category_names)}})   