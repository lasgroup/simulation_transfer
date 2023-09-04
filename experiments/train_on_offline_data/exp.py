from experiments.data_provider import provide_data_and_sim


x_train, y_train, x_test, y_test, sim = provide_data_and_sim(data_source='real_racecar_new_actionstack',
                                                             data_spec={'num_samples_train': 10000})
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)




