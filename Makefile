IMG_NAME=mlapi

TAG=latest
ENV_TAG=latest

build-image:
	docker build --progress=plain --no-cache --rm -t ${IMG_NAME}:${TAG} .
	docker tag ${IMG_NAME}:${TAG} ${IMG_NAME}:${ENV_TAG}
