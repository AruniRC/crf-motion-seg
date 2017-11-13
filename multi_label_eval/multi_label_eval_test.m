
clear; clc;

data= [0.8; 0.1];

%data = load('F_cars9.mat');

[mean, idx_table, n_values] = multi_label_eval(data, 'threshold', .75);
   
fprintf('measure_mean         : %f\n', mean);
fprintf('values above treshold: %02d\n', n_values);
fprintf('selected_idx         : %04d\n', idx_table);