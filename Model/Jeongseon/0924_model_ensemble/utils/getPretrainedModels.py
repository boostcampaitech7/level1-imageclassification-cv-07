import timm
#timm에서 제공하는 모든 pretrained model들을 가져와서 txt 파일로 저장
model_names = timm.list_models(pretrained=True)

f = open("pretrained_models.txt", 'w')

for model_name in model_names:
    f.write(model_name + "\n")