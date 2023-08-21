sudo ./bin/tritonserver --strict-model-config=false --allow-metrics=true --model-repository ./triton-deploy/models --backend-directory=./backends --model-control-mode=poll
