%Learns a 2-layer basic neural net.Parameterize number of hidden units and
%regularization
function [W1,W2] = trainneuralnet(X,Y,nhidden,lambda)
gridX = getgridpts(X);
%append 1's to x
X_1 = horzcat(ones(size(X,1),1),X);

%starting weights
W_1 = -1+(2).*rand(nhidden,size(X_1,2));% dimensions of first weight matrix 5x3, 10x3, 15x3
W_f = -1+(2).*rand(1, nhidden+1);%dimensions of second matrix 1x6, 1x16, 


i = 1; %iteration count
max_grad = 1;
while (max_grad > 0.001)
    grad_wf = zeros(size(W_f)); grad_w1=zeros(size(W_1));
    loss = 0; f_x =0;
    for row = 1:size(X_1,1)%for every row of X_1
        x_r = X_1(row,:);%extract ith row

        [z,f] = forwardprop(x_r,W_1,W_f);
        f_x(row,1) = f; %hypothesis for the ithe example
        [del1, del2] = backprop(z',f,Y(row),W_f);%transposing z to send as col vector

        delWf_L = del1 .* z; % for the last layer weights
        delW1_L = del2(2:end) * x_r;

        %accumulate gradients-running sum
        grad_wf = grad_wf + delWf_L;
        grad_w1 = grad_w1 + delW1_L;
    end
            
 %stopping criteria           
     delW_1 = grad_w1 + (2*lambda).*W_1;       
     delW_f = grad_wf + (2*lambda).*W_f;
 
  max_grad = max(abs([delW_1(:); delW_f(:)]));
          if(mod(i,1000) == 0)
              l = arrayfun(@losscum,f_x,Y);
              loss = sum(l);
              disp(['Loss =' num2str(loss)]);
              max_grad            
          end
i = i + 1;
eta = 1000/(25000+i);

%weight update
 W_1 =  W_1 - eta.*(delW_1);
 W_f =  W_f - eta.*(delW_f);
end
%classifier
        y_pred = zeros(size(Y));
        %based on the weights retured calculate y's
        X_f = horzcat(ones(size(gridX,1),1),gridX);
        for i = 1:size(X_f,1) %for every row
           xf_r = X_f(i,:);
           [~,f_xi] = forwardprop(xf_r,W_1,W_f); %f_xi contains hypothesis
           y_pred(i) = f_xi;
        end    
        plotdecision (X ,Y ,gridX , y_pred);
i    
W1 = W_1
W2 = W_f


