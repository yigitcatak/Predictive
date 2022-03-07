if not use_presplit:
    # read segmented data (segments fo Nloc) and shuffle samples

    # there is data imbalance between healthy class and others, and also between normal_1 and normal_2,3,4 data
    # check 'all_file_lengths.txt'
    # normal_1 has twice the data of anomalies, normal_2,3,4 has twice the data of normal_1
    # so there are 3.5 times more data for healthy class than others

    x_anomaly_test = []
    x_normal_train = []
    x_normal_test = []
    x_mixed_train = []
    x_mixed_test = []
    y_mixed_train = []
    y_mixed_test = []
    
    x_anomaly_train = []
    y_anomaly_train = []
    y_anomaly_test = []
    
    df = pd.DataFrame()
    working_condition = 0 #to be used to track the 4 working conditions of same class
    for name in all_files:
        df = df.append(pd.read_csv(f'dataset/segmented/{name}'), ignore_index=True)
        working_condition += 1 
        if working_condition == 4: # when all the working conditions of the same class is collected shuffle it and add to the general list
            working_condition = 0
            data = GroupSamples(df.drop(['label'], axis=1).values,J)

            # get every Jth label as the labels are given with the average of J segments to classifier
            label = df['label'].values.tolist()[:len(data)*J:J]

            train, test, y_train, y_test = train_test_split(data,label,test_size=0.75)

            # unzip the samples as the samples are shuffled keeping segment order
            train = Flatten(train)
            test = Flatten(test)

            if 'normal' in name: # healthy data to train autoencoder
                x_normal_train += train
                x_normal_test += test

            else: # all anomaly data to check autoencoder error
                #idx = int(len(train)/2)
                #idx = idx - (idx%J)
                #train = train + train + train + train[:idx]
                #y_train = y_train + y_train + y_train + y_train[:int(idx/24)]
                x_anomaly_train += train
                x_anomaly_test += test

            # mixed data and label to train and test classifier
            x_mixed_train += train
            x_mixed_test += test
            y_mixed_train += y_train
            y_mixed_test += y_test

            df = pd.DataFrame()

    x_mixed_train, mean, eigenVecs, diagonal_mat = Whiten(x_mixed_train)
    x_mixed_test = Whiten(x_mixed_test, mean, eigenVecs, diagonal_mat)
    x_normal_train = Whiten(x_normal_train, mean, eigenVecs, diagonal_mat)
    x_normal_test = Whiten(x_normal_test, mean, eigenVecs, diagonal_mat)
    x_anomaly_train = Whiten(x_anomaly_train, mean, eigenVecs, diagonal_mat)
    x_anomaly_test = Whiten(x_anomaly_test, mean, eigenVecs, diagonal_mat)
    

    #grouped_normal = torch.tensor(GroupSamples(x_normal_train,J),dtype=torch.float32)
    #dfgrouped_mixed = torch.tensor(GroupSamples(x_mixed_train,J),dtype=torch.float32)

    x_anomaly_train = torch.tensor(x_anomaly_train,dtype=torch.float32)
    x_anomaly_test = torch.tensor(x_anomaly_test,dtype=torch.float32)
    x_normal_train = torch.tensor(x_normal_train,dtype=torch.float32)
    x_normal_test = torch.tensor(x_normal_test,dtype=torch.float32)
    x_mixed_train = torch.tensor(x_mixed_train,dtype=torch.float32)
    x_mixed_test = torch.tensor(x_mixed_test,dtype=torch.float32)
    
    y_mixed_train = torch.tensor(y_mixed_train,dtype=torch.long)
    y_mixed_test = torch.tensor(y_mixed_test,dtype=torch.long)

    if update_presplit:
        torch.save(x_anomaly_train,'dataset/presplit/x_anomaly_train.pt')
        torch.save(x_anomaly_test,'dataset/presplit/x_anomaly_test.pt')
        torch.save(x_normal_train,'dataset/presplit/x_anomaly_train.pt')
        torch.save(x_normal_test,'dataset/presplit/x_normal_test.pt')
        torch.save(x_mixed_train,'dataset/presplit/x_mixed_train.pt')
        torch.save(x_mixed_test,'dataset/presplit/x_mixed_test.pt')
        torch.save(y_mixed_train,'dataset/presplit/y_mixed_train.pt')
        torch.save(y_mixed_test,'dataset/presplit/y_mixed_test.pt')

    del df, working_condition, train, test, temp

else: #use pre-split data
    x_mixed_train = torch.load('dataset/presplit/x_mixed_train.pt')
    x_mixed_test = torch.load('dataset/presplit/x_mixed_test.pt')
    x_normal_train = torch.load('dataset/presplit/x_normal_train.pt')
    x_normal_test = torch.load('dataset/presplit/x_normal_test.pt')
    x_anomaly_train = torch.load('dataset/presplit/x_anomaly_test.pt')
    x_anomaly_test = torch.load('dataset/presplit/x_anomaly_test.pt')
    y_mixed_train = torch.load('dataset/presplit/y_mixed_train.pt')
    y_mixed_test = torch.load('dataset/presplit/y_mixed_test.pt')

input_dim = x_mixed_train.shape[1]