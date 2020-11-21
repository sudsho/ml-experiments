# tf serving exploration. just notes for now.
# plan: export the tf2 mnist model as a savedmodel, run tf serving in docker,
# hit it with a curl POST.
import tensorflow as tf

# placeholder; the real export goes here once the model is trained.
# model.save('saved_model/mnist/1/')

# docker run -p 8501:8501 \
#   --mount type=bind,source=$(pwd)/saved_model/mnist,target=/models/mnist \
#   -e MODEL_NAME=mnist tensorflow/serving

# curl -d '{"instances": [[...]]}' \
#   -X POST http://localhost:8501/v1/models/mnist:predict
