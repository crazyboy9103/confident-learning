# confident-learning
```
docker build -t ${IMAGE_TAG} .
docker run -it --name ${NAME} --gpus all --ipc host --network host -v ${HOST_DATA_PATH}:/datasets -v ${HOST_PROJECT_PATH}:/workspace ${IMAGE_TAG}
```

# custom dataset 
dataset arg must be imagefolder

for folder structure for custom dataset, refer to https://huggingface.co/docs/datasets/create_dataset#folder-based-builders

~/.cache/huggingface/datasets is used to store datasets fetched from huggingface hub 