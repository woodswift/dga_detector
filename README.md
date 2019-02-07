# dga_detector

1. **dga-dataset.txt**: the raw data;
2. **rnn_clf.py**: the script which completes processing data, training model and evaluating performance;
3. **test_api.py**: the script which shows how to use the micro-service;
4. **development_summary.txt**: the report summarizing what have been explored and developed in this project;
5. **api_instruction.txt**: the instruction introducing where to find the docker image and how to use the micro-service;
6. **models folder**: serialized model objects;
7. **micro-service folder**: all scripts, files and objects needed for the micro-service;

The data was split into train, dev and test datasets, and RNN algorithm was applied to learn the pattern of character dependency.

The precision, recall and f1-score achieved on the test dataset are all nearly close to 100%, and AUC of the model is greater than 0.9999. Please find more details in **development_summary.txt**.
