# confident-learning
```
docker build -t ${IMAGE_TAG} .
docker run -it --name ${NAME} --gpus all --ipc host --network host -v ${HOST_DATA_PATH}:/datasets -v ${HOST_PROJECT_PATH}:/workspace ${IMAGE_TAG}
```