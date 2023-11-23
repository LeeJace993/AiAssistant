import pickle


def trainSVM():
    model_pkl_filename='utils/prediction_test_classifier.pkl'
    with open(model_pkl_filename,'rb') as model_pkl:
        clf=pickle.load(model_pkl)
    return clf

    
    
    
    
    
   
    
    

    
    
    
    
    
    
    
    
    