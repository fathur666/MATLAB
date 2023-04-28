clear; %Clear Workspace
clc; % Clear Command Line
close all; %Close all plot and window

%INISIALISASI DATA SET
allData = importdata('wdbc.csv');
%AMBIL LABEL
get_unique = unique(allData.textdata(:,2));
%ONE HOT
Y_label = zeros(569,2);
for i = 1:2
    Y_label(:,i) = contains(allData.textdata(:,2),get_unique{i});
end

%NORMALISASI DATA DENGAN MIN-MAX SCALER DARI -5 SAMPAI 5
X = allData.data;
a = -5;
b = 5;
X_train_min = min(X);
X_train_max = max(X);

for j = 1:30
    for i = 1:569
        X_aksen(i,j) = a + ((X(i,j)-X_train_min(1,j))*(b-a))/(X_train_max(1,j)-X_train_min(1,j));
    end
end

%NORMALISASI DENGAN Z-NORMALIZATION
rata_rata = mean(X_aksen);
simpangan_baku = std(X_aksen);

for j = 1:30
    for i = 1:569
        Z(i,j) = (X_aksen(i,j)-rata_rata)/simpangan_baku;
    end
end

%CEK EIGEN VALUE
cova = cov(Z);
[V,D] = eig(cova);

%MEMBAGI DATA TRAINING 60% DAN TESTING 40%
X_train = Z(1:341,:);
X_test = Z(342:end,:);
Y_train = Y_label(1:341,:);
Y_test = Y_label(342:end,:);

n = 30; %INPUT
p = 20; %HIDDEN LAYER
m = 2; %OUTPUT
v_ij = rand(n,p) - 0.5; %BOBOT INPUT HIDDEN
w_jk = rand(p,m) - 0.5; %BOBOT HIDDEN OUTPUT
beta_v = 0.7*(p).^(1/n); %BETA V
beta_w = 0.7*(m).^(1/p); %BETA W
v_0j = rand(1,p).*(2.*beta_v) - beta_v; %BIAS INPUT HIDDEN
w_0k = rand(1,m).*(2.*beta_w) - beta_w; %BIAS HIDDEN OUTPUT

norm_v = sqrt(sum(v_ij.^2,'all'));
norm_w = sqrt(sum(w_jk.^2,'all'));
%NGUYEN WIDROW
for i = 1:n
    for j = 1:p
        v_ij(i,j) = (beta_v*v_ij(i,j))/norm_v; %UPDATE BOBOT TERBARU
    end
end

for i = 1:p
    for j = 1:m
        w_jk(i,j) = (beta_w*w_jk(i,j))/norm_w; %UPDATE BOBOT
    end
end

epoch_maksimum = 10000; %Epoch Maksimum
error_stop = 0.01; %Stop Error Minimum

alpha = 0.2; %Learning Rate Awal
miu = 0.7; %Momentum Awal
stop_toggle = 0; %Toggle Buat Stop
epoch = 1; %Epoch Awal
delta_wjk_old = 0; %Memori Perubahan Bobot Input Hidden
delta_w0k_old = 0; %Memori Perubahan Bias Input Hidden
delta_vij_old = 0; %Memori Perubahan Bobot Hidden Output
delta_v0j_old = 0; %Memori Perubahan Bias Hidden Output
errorperepoch = zeros(1,epoch_maksimum); %Hitung Error Per epoch

%UBAH BOBOT
while stop_toggle == 0 && epoch <= epoch_maksimum
    for a=1:length(X_train)
        % Feedforward 
        xi = X_train(a,:);
        ti = Y_train(a,:);
        % Input layer -> hidden layer
        z_inj = xi*v_ij + v_0j;
        % Activation function with Sigmoid
        zj = zeros(1,p);
        for j=1:p
            zj(1,j) = 1/(1+exp(-z_inj(1,j))); 
        end
        % Hidden layer -> output layer
        y_ink = zj*w_jk + w_0k;
        yk = zeros(1,m);
        for k=1:m
            yk(1,k) = 1/(1+exp(-y_ink(1,k)));
        end
        % Simpan Error
        error(1,a) = 0.5*sum((ti - yk).^2);
        
        % Backpropagation of error %
        % Komputasi dari output layer ke hidden layer
        dok = ((ti - yk)).*(1-yk.^2);
        delta_wjk = alpha*zj'*dok + miu*delta_wjk_old;
        delta_w0k = alpha*dok + miu*delta_w0k_old;
        delta_wjk_old = delta_wjk;
        delta_w0k_old = delta_w0k;
        % Komputasi dari hidden layer ke input layer
        doinj = dok*w_jk';
        doj = doinj.*(1-zj.^2);
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
    
    if errorperepoch(1,epoch) < error_stop
        stop_toggle = 1;    %Error minima telah terpenuhi, berhenti
    end
    
    epoch = epoch+1; %Tambahkan Epochnya
end

jumlah_benar = 0;
jumlah_salah = 0;
Hasil_prediksi = zeros(length(X_test),m);
error_test = zeros(1,length(X_test));
y_test_all = zeros(length(X_test),2);

for a = 1:length(X_test)
    xi_test = X_test(a,:);
    ti_test = Y_test(a,:);
    % Input layer -> hidden layer
    z_inj_test = xi_test*v_ij + v_0j;      
    for j=1:p
        zj_test(1,j) = 1/(1+exp(-z_inj_test(1,j)));
    end
    % Hidden layer -> output layer
    y_ink_test = zj_test*w_jk + w_0k;
    for k=1:m
        yk_test(1,k) = 1/(1+exp(-y_ink_test(1,k)));
    end
    for j = 1:m
        Hasil_prediksi(a,j)=yk_test(j);
    end
    % Simpan Error
    error_test(1,a) = 0.5*sum((ti_test - yk_test).^2);
    % Menghitung recognition rate
    [value, index] = max(abs(yk_test));
    y_test = zeros(size(ti_test));
    y_test(1,index) = 1;
    y_test_all(a,:) = y_test;
    if y_test == ti_test
        jumlah_benar = jumlah_benar + 1;
    else
        jumlah_salah = jumlah_salah + 1;
    end
end

avgerrortest = sum(error_test)/length(X_test);
recog_rate = (jumlah_benar/length(X_test))*100;
test_length = length(X_train);
test_length = test_length - 1;
figure;
plot(errorperepoch);
ylabel('Value of Error per epoch'); xlabel('Epoch')
disp("Recognition rate = "+ recog_rate +" %");
disp("Jumlah Benar = "+ jumlah_benar + "/" + length(X_test));
disp("Dengan epoch = " + epoch + " dan error " + errorperepoch(1,epoch-1));
disp("Deviasi rata-rata test = "+ avgerrortest);