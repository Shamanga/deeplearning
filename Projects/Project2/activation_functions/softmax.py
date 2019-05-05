class Softmax(Module):
    
    def forward(self, *input):
        
        self.input_from_layer = input[0]
        
        return self.softmax(input_from_layer)
    
    def backward(self, *gradwrtoutput):
        
        derivatives = self.softmax_p(self.input_from_layer)
        
        return derivatives * gradwrtoutput[0]
        
    
    def softmax(self, input_to_compute):
        
        input_to_compute_v = input_to_compute.view(-1,1)
        
        norm_value = input_to_compute_v.max()
        
        stable_input_to_compute_v = input_to_compute_v - norm_value
        
        exponentials = torch.exp(stable_input_to_compute_v)
        
        sum_exponentials = torch.sum(exponentials)
        
        return (exponentials/sum_exponentials).view(-1)
    
    def softmax_p(self, input_to_compute_p):
        
        softmax_res = self.softmax(input_to_compute_p)
        
        diag_softm = torch.diag(softmax_res)
        
        derivative_soft = torch.FloatTensor(diag_softm.shape[0], diag_softm.shape[0])
        
        for i in range((diag_softm.shape[0])):
            for j in range((diag_softm.shape[0])):
                if i == j:
                    derivative_soft[i][j] = softmax_res[i] * (1 - softmax_res[i])
                else:
                    derivative_soft[i][j] = -softmax_res[i] * softmax_res[j]
                    
        return derivative_soft
    