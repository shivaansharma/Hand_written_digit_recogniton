import numpy as np
from Model import initialize_parameters,train,predict,xtrain,ytrain,load_weights

layers  = [784,256,128,10]

# W,B,l,e = initialize_parameters(layer_sizes=layers)
# train(W,B,l,e)
sample = xtrain[0]
sample = sample.reshape(1,-1)
W,B = load_weights()

lable,prob = predict(sample,W,B)

print("predlable :",lable)
print("actuallable :",np.argmax(ytrain[0]))
