./tao-converter/tao-converter ./model/lp/resnet10_180_90_belgium.etlt \
              -k nvidia_tlt \
              -o predictions/Softmax \
              -d 3,180,90 \
              -i nchw \
              -m 64 \
              -e ./triton-deploy/models/EULPNet/1/model.plan
