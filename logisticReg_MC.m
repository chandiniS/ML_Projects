%Multi class logistic regression  for classifying phising data
%has both linear and quadratic feature space
function[] = logistic(X,Y)
% ----------------Linear model for Logistic Regerssion---------
%calculate predictions- 1 fold cross validation
%split data in 80 : 20 ratio.(4:1)
disp('Starting logistic regression for Linear feature space with cross validation..');
train_size= 0.8*size(X,1);
TrainX = X(1:train_size,:);
TrainY = Y(1:train_size,:);

ValidX = X((train_size+1):end,:);
ValidY = Y((train_size+1):end,:);

TrainX_regul = horzcat(ones(size(TrainX,1),1),TrainX); %append ones
ValidX_regul = horzcat(ones(size(ValidX,1),1),ValidX);

lambdas = [0.01 0.1 1 10 100 1000];

lambda_err(:,1) = lambdas;
lambda_err(:,2) = zeros;

%run logistic regression on training data for different lambdas
    for l_indx = 1:size(lambdas,2)  
        w = learnlogreg(TrainX_regul,TrainY,lambdas(1,l_indx));%fitted weights
    %     disp(['weights for lambda=' num2str(lambdas(1,l_indx)) '=']);
    %     disp(w);
        mis_class = 0;
        %calculate error rates on the validation set
        for i= 1: size(ValidX_regul,1)%for each example in validation set
           %calculate predicted value using the sigmoid  - probability of class
           %is +ve for a given x = 1/(1+exp(-validX*w)
           X_W = 0; %clear variable before starting for each example
           for j = 1:size(ValidX_regul,2)%iterate over each column(feature of the example)
                X_W = X_W+ValidX_regul(i,j)*w(j,1);
           end
           h_i = 1/(1+exp(-(X_W)));%this should give number between 0 &1

           %error calculation
           if (h_i < 0.5)%prediction is negative
               if(ValidY(i,1)>0)%actual is positive
                   mis_class = mis_class+1;%increment error count
               end    
           else %predicting positive example    
                if(ValidY(i,1)<0)%actual is negative
                    mis_class = mis_class+1;%increment error count
                end    
           end   

        end
        %track lambda & error counts
%        disp(['Total number of misclassifications for lambda =' num2str(lambdas(l_indx)) ' is =' num2str(mis_class)]);
        lambda_err(l_indx,2) = (mis_class/size(ValidX_regul,1))*100;        
    end

%for linear models
disp('Lambda - err value(%) =');
format shortG;
disp(lambda_err);
clear Train_X Train_Y ValidX ValidY TrainX_regul ValidX_regul;

%------------------Linear Model with 10-fold cross validation---------------------
disp('Starting logistic regression for Linear feature space with 10-fold cross validation..(takes 2-3min)');
%matrix to track cv err per lambda
lambdas_cv = [0.1 1 10 100 1000];
lambda_cverr(:,1) = lambdas_cv;
lambda_cverr(:,2) = zeros;
%choosing n-fold as 10 fold
nfold = 10;
num_valid = size(X,1) /nfold;
X_r = horzcat(ones(size(X,1),1),X);

for l_indx = 1:size(lambdas_cv,2) 
    avg_misfold = 0;%variable to track error for each lambda as a average of 10 folds
    for n = 0:nfold-1 %for each fold
       st_idx = num_valid*n +1; 
       end_idx = st_idx+(num_valid -1);
       
       valid_X = X_r([st_idx : end_idx],:); 
       valid_Y = Y([st_idx : end_idx],:);
       tr_x = X_r; tr_y = Y;
       tr_x(([st_idx : end_idx]),:)=[];
       tr_y(([st_idx : end_idx]),:)=[];
       
       w_cv = learnlogreg(tr_x,tr_y,lambdas_cv(1,l_indx));
       
       mis_fold = 0;
       for i= 1: size(valid_X,1)%for each example in validation set
           %calculate predicted value using the sigmoid  - probability of class
           %is +ve for a given x = 1/(1+exp(-validX*w)
           X_W = 0; %clear variable before starting for each example
           for j = 1:size(valid_X,2)%iterate over each column(feature of the example)
                X_W = X_W+valid_X(i,j)*w_cv(j,1);
           end
           h_i = 1/(1+exp(-(X_W)));%this should give number between 0 &1

           %error calculation
           if (h_i < 0.5)%prediction is negative
               if(valid_Y(i,1)>0)%actual is positive
                   mis_fold = mis_fold+1;%increment error count
               end    
           else %predicting positive example    
                if(valid_Y(i,1)<0)%actual is negative
                    mis_fold = mis_fold+1;%increment error count
                end    
           end   
       end
       avg_misfold = avg_misfold+mis_fold;%to keep accumulate errors per fold     
    end
    %store per lambda the avg of errors across all folds divided by
    %validationset size and converted to percentage
    lambda_cverr(l_indx,2) = ( avg_misfold /(10 * size(valid_X,1)))*100; 
    
end    
    disp('Lambda - (10-fold CV)err value(%) =');
    format shortG;
    disp(lambda_cverr);
clear tr_x tr_y valid_X valid_Y X_r;
    


%------------------Quadratic Model for logistic regression-----------------
%do quadratic fitting
%augment the test data
   aug_X = X;
    for i = 1:size(X,2)
        for j = 1:size(X,2)
            aug_X = horzcat(aug_X,aug_X(:,i).*aug_X(:,j));
        end
    end    
    new_x = unique(aug_X','rows','stable');
    aug_X = new_x';
%augmenting done

disp('Starting logistic regression for Quadratic feature space with simple cross validation..');
    %split data into training and validation
    train_size= 0.8*size(aug_X,1);
    QTrainX = aug_X(1:train_size,:);
    QTrainY = Y(1:train_size,:);
   
    QValidX = aug_X((train_size+1):end,:);
    QValidY = Y((train_size+1):end,:);
    
    %append ones to regularize
    QTrainX_r = horzcat(ones(size(QTrainX,1),1),QTrainX); 
    QValidX_r = horzcat(ones(size(QValidX,1),1),QValidX);
   
    lambdas = [0.1 1 10 100 1000];
    lambda_qerr(:,1) = lambdas;
    lambda_qerr(:,2) = zeros;
    
    for indx = 1:size(lambdas,2)
        w_q = learnlogreg(QTrainX_r,QTrainY,lambdas(1,indx));%fitted weights fro quadratic model
    %     disp(['weights for lambda=' num2str(lambdas(1,l_indx)) '=']);
    %     disp(w);
        qmis_class = 0;
        %calculate error rates on the validation set
        for i= 1: size(QValidX_r,1)%for each example in validation set
           X_W = 0; %clear variable before starting for each example
           for j = 1:size(QValidX_r,2)%iterate over each column(feature of the example)
                X_W = X_W+QValidX_r(i,j)*w_q(j,1);
           end
           h_i = 1/(1+exp(-(X_W)));%this should give number between 0 &1

           %error calculation
           if (h_i < 0.5)%prediction is negative
               if(QValidY(i,1)>0)%actual is positive
                   qmis_class = qmis_class+1;%increment error count
               end    
           else %predicting positive example    
                if(QValidY(i,1)<0)%actual is negative
                   qmis_class = qmis_class+1;%increment error count
                end    
           end   

        end
        %track lambda & error counts
       
       lambda_qerr(indx,2) = (qmis_class/size(QValidX_r,1))*100;       
    end
    disp('Lambda - quadratic err value(%) =');
    format shortG;
    disp(lambda_qerr);
    clear QTrainX QTrainY QValidX QValidY QTrainX_r QValid_r;
    
%------------Logistic Regression in quadratic feature space with 10-fold---
%************Commenting rhis code since it is taking too long**************
%     disp('Starting logistic regression for Quadratic feature space with 10-fold cross validation..');
% 
%     lambdas_cv = [0.1 1 10 100 1000];
%     lambda_cvqerr(:,1) = lambdas;
%     lambda_cvqerr(:,2) = zeros;
%     
% 
%     augX_r = horzcat(ones(size(aug_X,1),1),aug_X);
%     for l_indx = 1:size(lambdas_cv,2) 
%         avg_misfold = 0;%variable to track error for each lambda as a average of 10 folds
%         for n = 0:nfold-1 %for each fold
%            st_idx = num_valid*n +1; 
%            end_idx = st_idx+(num_valid -1);
% 
%            valid_X = augX_r([st_idx : end_idx],:); 
%            valid_Y = Y([st_idx : end_idx],:);
%            tr_x = augX_r; tr_y = Y;
%            tr_x(([st_idx : end_idx]),:)=[];
%            tr_y(([st_idx : end_idx]),:)=[];
% 
%            w_cv = learnlogreg(tr_x,tr_y,lambdas_cv(1,l_indx));
% 
%            mis_fold = 0;
%            for i= 1: size(valid_X,1)%for each example in validation set
%                %calculate predicted value using the sigmoid  - probability of class
%                %is +ve for a given x = 1/(1+exp(-validX*w)
%                X_W = 0; %clear variable before starting for each example
%                for j = 1:size(valid_X,2)%iterate over each column(feature of the example)
%                     X_W = X_W+valid_X(i,j)*w_cv(j,1);
%                end
%                h_i = 1/(1+exp(-(X_W)));%this should give number between 0 &1
% 
%                %error calculation
%                if (h_i < 0.5)%prediction is negative
%                    if(valid_Y(i,1)>0)%actual is positive
%                        mis_fold = mis_fold+1;%increment error count
%                    end    
%                else %predicting positive example    
%                     if(valid_Y(i,1)<0)%actual is negative
%                         mis_fold = mis_fold+1;%increment error count
%                     end    
%                end   
%            end
%            avg_misfold = avg_misfold+mis_fold;%to keep accumulate errors per fold     
%         end
%         %store per lambda the avg of errors across all folds divided by
%         %validationset size and converted to percentage
%         lambda_cverr(l_indx,2) = ( avg_misfold /(10 * size(valid_X,1)))*100; 
%     end
%     
%     disp('Lambda - (10-fold CV)quadratic-err value(%) =');
%     format shortG;
%     disp(lambda_cverr);
%     
%    clear tr_x tr_y valid_X valid_Y X_r;

end