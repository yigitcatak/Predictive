addpath('/Users/yigit/Documents/VSCode/Python/predictive/datasets/Paderborn/original');
names=dir('/Users/yigit/Documents/VSCode/Python/predictive/datasets/Paderborn/original');
Nloc=213;
var_names=cell(1,Nloc+1);
var_names{end}='label';
for i=1:Nloc
    var_names{i}=num2str(i-1);
end

filenames = ["K001","K002","K003","K004","K005","KA04","KA15","KA16","KA22","KA30","KI04","KI14","KI16","KI18","KI21"];
for j=1:length(filenames)
    temp_array = [];
    filename = filenames(j);
    flag = false;
    for i=3:length(names)
        tmp_name=names(i).name;
        tmp=load(tmp_name);
        tmp=struct2cell(tmp);
        tmp=tmp{1,1};
        if strcmp(tmp_name(13:14),'K0')
            label=0;
        elseif strcmp(tmp_name(13:14),'KI')
            label=1;
        elseif strcmp(tmp_name(13:14),'KA')
            label=2;
        else
            label=3;
        end
        segmented = segmentation(tmp.Y(7).Data,Nloc,label);
        % vibration 7, current1 2, current2 3
        
        if strcmp(tmp_name(13:16),filename)
            flag = true;
            temp_array = [temp_array; segmented];
        end
    end
    if flag
        temp_array = array2table(temp_array,'VariableNames',var_names);
        writetable(temp_array,['vibration_',convertStringsToChars(filename),'.csv'],'Delimiter',',','QuoteStrings',true)
    end
end

function segmented_data = segmentation(data,Nloc,label)
    total_iter=(length(data)-mod(length(data),Nloc))/Nloc;
    segmented_data=zeros(total_iter,Nloc);
    for i=1:total_iter
        segmented_data(i,:)=data((i-1)*Nloc+1:i*Nloc);
    end
    segmented_data(:,end+1)=label;
end