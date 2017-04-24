function [Y_hat] = SVMSpamClassify()
X= load('spamtrainX.data', '-ascii');

Y = load('spamtrainY.data','-ascii');

C = logspace(-2,3);
%for different values of C run learn svm & get the error computed
 
%do simple cross-validatio to begin with- split data 80:20
indx = 0.75*size(X,1);%Calculate 75th split of train data

X_train = X(1:indx,:);
Y_train = Y(1:indx,:);
X_valid = X(indx+1:end,:);
Y_valid = Y(indx+1:end,:);
C_errmat = horzcat(C',zeros(size(C,2),1));
 wmat_t = ones(1,size(X,2));% weights stored as transpose
 bmat =zeros(size(X,2),1);
for c_idx = 1:size(C,2)%for different C
    w=0;b=0;v_err=0;
    [w,b] = learnsvm(X_train,Y_train,C(c_idx));
    
    for j = 1: size(X_valid,1)
        h_w = 0;
        for k = 1:size(w,1)
            h_w = h_w+ X(j,k)*w(k,1);
        end    
        if(Y_valid(j)*(h_w+b) < 0)%prediction is incorrect
            v_err = v_err+1;
        end    
        
    end   
    
    C_errmat(c_idx,2) = v_err;
    wmat_t = vertcat(wmat_t,w');
    bmat(c_idx,1) = b;
end;    

wmat_t(1,:) =[];%not required ones
%get minimum C value
[minval, minidx] = min(C_errmat(:,2));
disp(['Best C =']);
bestC = C(1,minidx)
bestw = wmat_t(minidx,:);
bestb = bmat(minidx);
Test_X=load('spamtestX.data', '-ascii');
Y_hat = zeros(size(Test_X,1),1);

    for n =  1:size(Test_X)
        x_t = Test_X(n,:); %row of test data

        if( (x_t * bestw' + bestb) > 0)
            Y_hat(n) = +1;
        else
            Y_hat(n) = -1;
        end    

    end    

end