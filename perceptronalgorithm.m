%Implements the perceptron algorithm. n is number of sweeps, eta is learning rate
% Try this on linearly seperable data
function w = learnperceptron(X,Y,n,eta,w0,animate)

close all;
figure;
hold on;
axis([-3,2,-3,2])
%Setup the data to be plotted -scatter points
plotdata = horzcat(X,Y);
criteria = plotdata(:,end) == 1;
pos_blue = plotdata(criteria,:);
neg_red = plotdata(not(criteria),:);
%split data based on class for plotting
    for indx = 1:size(pos_blue,1)
   
        plot(pos_blue(indx,end-2),pos_blue(indx,end-1),'x','color','b');
    end
    for indx = 1:size(neg_red,1)
        
        plot(neg_red(indx,end-2),neg_red(indx,end-1),'o','color','r');
    end
    
h_line = 0;% initialize line handle


%Add 1's to X because it is simpleprob.dat
new_X = horzcat(ones(size(X,1),1),X);
%last column in file is desired outpput


[row_x,col_x] = size(new_X);%note this x will includes 1's column

%Intialize w to whatever comes from parameters
w = w0;% weights are a column vector- should be of same length as X's col dimension +1
w_change = w0; % to track changes in the weight vector
w_sweep =w0;
%Initialize eta & sweep to begin with

sweep = 1;

% for each point in x- calculate the hypotheses-h= xo.wo + x1.w1 + x2.w2
%check if hypothese is >0 or <0   based on that assign n as +1 or -1
%calculate the error= actual - hypothesis (n - h)
% corection  d = eta * e
%new weights wo = wo+xo*d w1= w1+x1*d

    while(sweep<n)
        for i = 1:row_x %iterate over i'th data example
                h_i = 0;%initialize this to zero before each i's calc. to be safe
                for j = 1:col_x %iterate over each feature
                    h_i = h_i + new_X(i,j)*w(j,1);%calculate hypothesis for each i'th example             
                end

            if(Y(i)*h_i < 0) % hypotheses is wrong
                %update weights
                for k = 1:size(w,1)%for each element of w
                    w(k) = w(k) + (eta*Y(i,1)*new_X(i,k));
                end   

                %append new weights to change vector
                w_change = horzcat(w_change,w);

                %do animation here
                if(animate == 1)
                    if(h_line == -1)
                        disp('line is out of plot area');
                       break; 
                    end    
                    if(h_line ~= 0 ) %previous line  is present
%                         set(h_line,'visible','off');
                        delete(h_line);%clear it                    
                    end    
                    h_line =drawline(w(2:end), w(1));
                     pause (0.2);
                end 

            end 
            
        end  
     w_sweep = horzcat(w_sweep,w);  %at the end of each sweep store weights
     tmp = w_sweep(:,end) -w_sweep(:,end-1);
     if(sum(tmp) ==0)
         break;% done
     else    
        sweep = sweep+1; 
     end   
    
 
%Criteria to stop the 
    end    
%    disp(w_change);
   disp(['Sweep count=' num2str(sweep)]);
   final_w = w_sweep(:,end);
   h =drawline(final_w(2:end), final_w(1));
   h.LineWidth =2;
end