import numpy as np
#import numpy as npr

class ImEx:
    def __init__(self,s,im_A,ex_A,im_B,ex_B,emb_B=None,im_C=None,ex_C=None):
        self.s = s
        self.im_A = np.array(im_A)
        self.ex_A = np.array(ex_A)
        
        self.im_B = np.array(im_B)
        self.ex_B = np.array(ex_B)
        
        self.im_C = np.array(im_C)
        self.ex_C = np.array(ex_C)
        self.emb_B = np.array(emb_B)
        
        self.r_of_stb = 0.0
        
        #print(im_A,im_B)
        #print(ex_A,ex_B)
        
    def cal_reg_of_stab(self):
        self.r_of_stb = 1.2+1j*1.2
        return(self.r_of_stb)
