function [y, mod_feature_train] = extract_features_and_labels(filename, total_r_entries)
    fileID = fopen(filename);
    formatSpec = '%f %f %f';
    sizeA = [total_r_entries, inf];
    features = fscanf(fileID, formatSpec, sizeA);
    features = transpose(features); 
    mod_feature_train = [];
    for i = 1:length(features)
        if(features(i, 1) == 2 || features(i, 1) == 8)
            mod_feature_train = [mod_feature_train, 
                                    features(i, :)]; 
        end
    end
    y = mod_feature_train(:,1);
    mod_feature_train(:,1) = []; 
end 


