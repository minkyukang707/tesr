from tflite_runtime.interpreter import Interpreter
import tqdm
import numpy as np
import pandas as p

model_path='/home/pi/tesr/lstm975_ae_0802.tflite'    

if __name__=='__main__':
    
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    interpreter.invoke()
    
