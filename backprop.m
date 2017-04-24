function[del1,del2] = backprop(z,f,y,W_2)
        del1 = f -y;
        
        tmp = del1.*W_2;
        del2 = tmp'.*arrayfun(@deriv,z);
        

end

function [z_d] = deriv(z)
    z_d = z*(1-z);

end
