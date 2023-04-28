clear; %Clear Workspace
clc; % Clear Command Line
close all; %Close all plot and window

% Inisialisasi dataset
dataset = readtable('diabetes_data_upload.csv','PreserveVariableNames',true);
N = size(dataset, 1);
% Mengambil jenis attribut yang ada pada dataset
fitur = dataset.Properties.VariableNames;
data = table2cell(dataset);

% Shuffle Data
data = data(randperm(N),:);

% Data Output Features (Class)
Y = data(:,17);
label = unique(Y);

% One Hot Encoding
Y_label = zeros(520,2);
for i = 1:2
    Y_label(:,i) = contains(Y,label(i));
end

% Data Input Features (Attribute)
age = data(:,1);
age = cell2mat(age);
gender = data(:,2);
gender = double(contains(gender,'Male'));
data(:,[1,2,17])=[];
X = double(contains(data,'Yes'));
X = [age gender X];

% Normalisasi data dengan metode z-score
X_z = zscore(X);

% Membagi data menjadi 80% train data dan 20% test data
X_train = X_z(1:0.7*N,:);
X_test = X_z(0.7*N+1:N,:);
Y_train = Y_label(1:0.7*N,:);
Y_test = Y_label(0.7*N+1:N,:);

% Create ANN Model
%1) Inisialisasi Data & Bobot
n = length(X(1,:));                     % Jumlah Input Layer
p = 13;                                 % Jumlah Hidden Layer    
m = 2;                                  % Jumlah Output Layer
v_ij = rand(n,p) - 0.5;                 % Bobot Input->Hidden
w_jk = rand(p,m) - 0.5;                 % Bobot Hidden->Output
beta_v = 0.7*(p).^(1/n);                % Bobot Nguyen Widrow V
beta_w = 0.7*(m).^(1/p);                % Bobot Nguyen Widrow W
v_0j = rand(1,p).*(2.*beta_v) - beta_v; % Bias Input->Hidden
w_0k = rand(1,m).*(2.*beta_w) - beta_w; % Bias Hidden->Output

norm_v = sqrt(sum(v_ij.^2,'all'));      % Normalisasi Bobot V
norm_w = sqrt(sum(w_jk.^2,'all'));      % Normalisasi Bobot W

for i = 1:n
    for j = 1:p
        v_ij(i,j) = (beta_v*v_ij(i,j))/norm_v; % Update Bobot Terbaru V
    end
end

for i = 1:p
    for j = 1:m
        w_jk(i,j) = (beta_w*w_jk(i,j))/norm_w; % Update Bobot Terbaru W
    end
end

alpha = 0.2;             % Learning Rate 
miu = 0.8;               % Momentum 
stop_flag = 0;         % Toggle untuk stop
epoch = 1;               % Epoch awal

error_min = 0.0001;       % Stop error minimum
epoch_maksimum = 100; % Epoch maksimum
errorperepoch = zeros(1,epoch_maksimum); 

delta_wjk_old = 0; %Memori Perubahan Bobot Input Hidden
delta_w0k_old = 0; %Memori Perubahan Bias Input Hidden
delta_vij_old = 0; %Memori Perubahan Bobot Hidden Output
delta_v0j_old = 0; %Memori Perubahan Bias Hidden Output

zj = zeros(1,p);
yk = zeros(1,m);

%2) Melatih Data
% Pengubahan bobot setiap input data
while stop_flag == 0 && epoch < epoch_maksimum
    % Proses Feedforward 
    for i=1:length(X_train)
        % Menerima Input dan mengirimkan ke hidden layer
        xi = X_train(i,:);
        z_inj = xi*v_ij + v_0j;
        
        % Menghitung nilai aktivasi setiap data input
        for j=1:p
            zj(1,j) = 1/(1+exp(-z_inj(1,j))); 
        end
        % Menghitung Nilai Output
        y_ink = zj*w_jk + w_0k;
        % Menghitung nilai aktivasi setiap data output
        
        for k=1:m
            yk(1,k) = 1/(1+exp(-y_ink(1,k))); 
        end
        % Simpan Error (Quadratic Error/MSE)
        yi = Y_train(i,:);
        error(1,i) = (1/length(X_test))*sum((yi - yk).^2);
        
        % Backpropagation of error 
        % Komputasi dari output layer ke hidden layer
        % Menghitung informasi error output
        dok = ((yi - yk)).*(yk-yk.^2);
        % Menghitung koreksi bobot unit output dan biasnya
        delta_wjk = alpha*zj'*dok + miu*delta_wjk_old;
        delta_w0k = alpha*dok + miu*delta_w0k_old;
        delta_wjk_old = delta_wjk;
        delta_w0k_old = delta_w0k;
        % Komputasi dari hidden layer ke input layer
        % Menghitung informasi error hidden
        doinj = dok*w_jk';
        doj = doinj.*(zj-zj.^2);
        % Menghitung koreksi bobot unit hidden dan biasnya
        delta_vij = alpha*xi'*doj + miu*delta_vij_old;
        delta_v0j = alpha*doj + miu*delta_v0j_old;
        delta_vij_old = delta_vij;
        delta_v0j_old = delta_v0j;
        % Memperbarui bobot dan bias
        w_jk = w_jk + delta_wjk;
        w_0k = w_0k + delta_w0k;
        v_ij = v_ij + delta_vij;
        v_0j = v_0j + delta_v0j; 
    end
    errorperepoch(1,epoch) = sum(error)/length(X_train);    
    
    if errorperepoch(1,epoch) < error_min
        stop_flag = 1;    %Jika mencapai error minimum, keluar loop
    end
    
    epoch = epoch+1;
end

%3) Tes Model
jumlah_benar = 0;
Hasil_prediksi = zeros(length(X_test),m);
error_test = zeros(1,length(X_test));
zj_test = zeros(1,p);
yk_test = zeros(1,m);
y_pred_all = zeros(length(X_test),2);

%Proses feedforward tanpa backpropagation
for i = 1:length(X_test)
    xi_test = X_test(i,:);
    yi_test = Y_test(i,:);
    z_inj_test = xi_test*v_ij + v_0j;
    
    for j=1:p
        zj_test(1,j) = 1/(1+exp(-z_inj_test(1,j)));
    end
    y_ink_test = zj_test*w_jk + w_0k;
    
    for k=1:m
        yk_test(1,k) = 1/(1+exp(-y_ink_test(1,k)));
    end
    for j = 1:m
        Hasil_prediksi(i,j)=yk_test(j);
    end
    % Simpan Error
    error_test(1,i) = (1/length(X_test))*sum((yi - yk).^2);
    % Menghitung recognition rate
    [value, index] = max(abs(yk_test));
    y_pred = zeros(size(yi_test));
    y_pred(1,index) = round(value);
    y_pred_all(i,:) = y_pred;
    if y_pred == yi_test
        jumlah_benar = jumlah_benar + 1;
    end
end

avgerrortest = sum(error_test)/length(X_test);
recog_rate = (jumlah_benar/length(X_test))*100;

%Plot Grafik Error Ketika Test vs Epoch
figure;
plot(errorperepoch);
ylabel('Training Error per epoch'); xlabel('Epoch')
disp("Recognition rate = "+ recog_rate +" %");
disp("Jumlah Benar = "+ jumlah_benar + "/" + length(X_test));
disp("Deviasi rata-rata test = "+ avgerrortest);
disp("Dengan epoch = " + epoch + " dan error saat training " + errorperepoch(:,epoch-1));