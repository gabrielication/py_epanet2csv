
close all;
clear all

figpath='figs/';
fontSize=12;
lineWidth=0.9;
markerSize=9;
marker={ 'o' 's' '*' 'd' '^' 's' '^' 's' '^' 'd' '.' 'd' 'p' 'x'};
style={'-' '--' '-.' ':'};

path='tensorflow_datasets/one_res_small/no_leaks_rand_base_demand/1M/';
nameFigure ={
    'nodes_position'
    };
saveFigEnable = 1;
files={
     %'1M_one_res_small_no_leaks_rand_bd_ordered_merged_link_reduced'
     '1M_one_res_small_no_leaks_rand_bd_ordered_new_delimited_merged'
}

data_raw_all = {};
for f=1:1
    filename_dat=[path files{f} '.csv'];
    fprintf('%s\n',filename_dat);
    %%% LOAD DATA %%%%%%%%%%%%%
    fid   = fopen(filename_dat);
    %line_ex = fgetl(fid);  % read line excluding newline character
    %disp(line_ex);
    %line_ex = fgetl(fid);  % read line excluding newline character
    %disp(line_ex);
    %0:00:00,8614,
    %0.00087812,0.00087812,53.84869719,35.41134519,494314.79000000,1377486.46000000,
    %Junction,False,
    %0.00000000,0.00000000,0.00000000,0.00087812,0.00000000,0.00087812,
    %[]
%     data_raw_cell = textscan(fid, '%s%s%f%f%f%f%f%f%s%s%f%f%f%f%f%f%q', 'Delimiter', ',', 'headerlines', 1);
    data_raw_cell = textscan(fid, '%s%s%f%f%f%f%f%f%s%s%f%f%f%f%f%f%q', 'Delimiter', ';', 'headerlines', 1);
    fclose(fid);

    for kk=1:length(data_raw_cell{1})
        %'['8600', '8604']'
        data_raw_cell{17}{kk} = strrep(data_raw_cell{17}{kk},'[','');
        data_raw_cell{17}{kk} = strrep(data_raw_cell{17}{kk},']','');
        if length(data_raw_cell{17}{kk})>0
            data_raw_cell{17}{kk} = strrep(data_raw_cell{17}{kk}, "'", "");
            links = strsplit(data_raw_cell{17}{kk}, ', ');
            for ll=1:length(links);
                data_raw_cell{18}{kk}{ll}=links{ll};
            end
        else
            data_raw_cell{18}{kk} = data_raw_cell{17}{kk};
        end
    end

    %data_raw = [];
    %for kk=1:length(data_raw_cell)-1
    %    data_raw = [data_raw data_raw_cell{kk}];
    %end
    %data_raw_all{f} = data_raw;
end

    
fig1=figure(1); 
set(gca, 'FontSize', fontSize);
leg1={};
gca1=gca;
gcf1=gcf;
hold on;

scatter(data_raw_cell{7}, data_raw_cell{8}, 20)

