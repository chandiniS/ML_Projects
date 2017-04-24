function[z,f] = forwardprop(x_r,W_1,W_2)
  
    a_1= W_1 * x_r'; %layer 1 activations
    z_1 = arrayfun(@sigmoid,a_1);%output of layer 1
  
    z_1 = vertcat(1,z_1);
 
    a_2 =  W_2 * z_1;% scalar
    
    f = sigmoid(a_2);
    z = z_1'; % return z as a transposed vector

end