import timm
model_names = timm.list_models(pretrained=True)

f = open("pretrained_models.txt", 'w')

for model_name in model_names:
    f.write(model_name + "\n")