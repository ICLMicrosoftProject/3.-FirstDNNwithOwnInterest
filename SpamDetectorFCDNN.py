import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cntk as C
import __future__, sys
import urllib
import zipfile
import csv
import re
##############Download the dataset
print("download our datasets")
urllib.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip", "/tmp/file.zip")

zip_ref = zipfile.ZipFile('/tmp/file.zip', 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()

##############read the downloaded csv file

exampleFile = open('/tmp/SMSSpamCollection')
exampleReader = csv.reader(exampleFile)
exampleData = list(exampleReader)

##############transform the file

for ii in range(0,len(exampleData)):
    if ii==0:
        y_raw_data=[re.sub(r'[^\w]', ' ',str(exampleData[ii]).split('\\t')[0])]
        x_raw_data=[re.sub(r'[^\w]', ' ',str(exampleData[ii]).split('\\t')[1])]
    else:
        y_raw_data= y_raw_data+[re.sub(r'[^\w]', ' ',str(exampleData[ii]).split('\\t')[0])]
        x_raw_data=x_raw_data+[re.sub(r'[^\w]', ' ',str(exampleData[ii]).split('\\t')[1])]


##############convert the format
#text=str(x_raw_data)##############convert the data into string format
def clean_and_split_data(text):
    text=text.lower()#################convert all character into lower case
    import re
    text_clean=re.sub(r'[^\w]', ' ', text)####remove all symbol
    text_clean=re.sub(r'[_]', ' ', text_clean)####remove special symbol
    text_clean=re.sub(r'[0-9]', '', text_clean)####remove all numbers
    words=text_clean.split()#####################split the sentence into words
    return(words)



for i in range(0,len(x_raw_data)):
    if i==0:
        words=clean_and_split_data(x_raw_data[i])
    else:
        words=words+clean_and_split_data(x_raw_data[i])


#########################################################################
###############################################Create a word vector list
#######################################################
word_text=np.unique(words)#####################find out the unique words
word_vec=np.eye(len(np.unique(words)))#########create a unique vector for each unique word
label_text=np.unique(y_raw_data)###############find out the unique label
label_vec=np.eye(len(np.unique(y_raw_data)))###create a unique vector for each unique label
##########################################################################

########################################################################
###############################################Convert input data and label into vectors
########################################################################
print("This may take some time for converting all sentences into vectors")
for i in xrange(0,len(x_raw_data)):
    print(i)
    print("convert sentence into vector out of")
    print(len(x_raw_data))
    temp_words=clean_and_split_data(x_raw_data[i])
    for ii in range(0,len(temp_words)):
        addr_word=word_text==temp_words[ii]#### find the address of words from the word vector list
        word_vec_temp=word_vec[addr_word]### extract the word vector
        if ii==0:
            sentence_vec=word_vec_temp
        else:
            sentence_vec=sentence_vec+word_vec_temp
    addr_label=label_text==y_raw_data[i]
    label_vec_temp=label_vec[addr_label]
    if i==0:
        x_text=[temp_words]
        x_vec=[sentence_vec[0]]
        y_text=[y_raw_data[i]]
        y_vec=[label_vec_temp[0]]
    else:
        x_text=x_text+[temp_words]
        x_vec=x_vec+[sentence_vec[0]]
        y_text=y_text+[y_raw_data[i]]
        y_vec=y_vec+[label_vec_temp[0]]
        
print('Now the input data is ready')
######################################################################################################
###############################################split the input data and label into training and testing
#######################################################################################################
print('we may split the data into training and spliting')
test_count= 574
train_count=len(x_raw_data)-test_count
training_x=x_vec[0:train_count]
testing_x=x_vec[train_count:(train_count+test_count)]
training_y=y_vec[0:train_count]
testing_y=y_vec[train_count:(train_count+test_count)]

#######################################################################################################
###############################################Batch the data set
#######################################################################################################
print('packaging our data into batchs, which in another word we aim to train our data batch by batch instead of one by one')
each_batch_size=50
for i in xrange(0,len(training_x)/each_batch_size):
    i_addr_start=i*each_batch_size
    i_addr_end=(i+1)*each_batch_size
    if i==0:
        training_batch_x=[training_x[i_addr_start:i_addr_end]]
        training_batch_y=[training_y[i_addr_start:i_addr_end]]
    else:
        training_batch_x=training_batch_x+[training_x[i_addr_start:i_addr_end]]
        training_batch_y=training_batch_y+[training_y[i_addr_start:i_addr_end]]


#######################################################################################################
print('now we are ready to train our dnn model')




# Ensure we always get the same amount of randomness
np.random.seed(0)
# Define the data dimensions
input_dim = len(training_batch_x[0][0])#7877
num_output_classes = 2
num_hidden_layers = 2
hidden_layers_dim = 100

input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)

def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]
    weight = C.parameter(shape=(input_dim, output_dim))
    bias = C.parameter(shape=(output_dim))
    return bias + C.times(input_var, weight)

def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)
    return nonlinearity(l)

# Define a multilayer feedforward classification model
def fully_connected_classifier_net(input_var, num_output_classes, hidden_layer_dim, 
                                   num_hidden_layers, nonlinearity):
    h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)
    
    return linear_layer(h, num_output_classes)


# Create the fully connected classfier
z = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim, 
                                   num_hidden_layers, C.sigmoid)

def create_model(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_output_classes, activation = None)
        
        return last_layer(h)
        

z = create_model(input)


loss = C.cross_entropy_with_softmax(z, label)
eval_error = C.classification_error(z, label)


# Instantiate the trainer object to drive the model training
learning_rate = 0.5
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch) 
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])

# Initialize the parameters for the trainer
minibatch_size = each_batch_size#50
num_samples = train_count#5000
num_minibatches_to_train = num_samples / minibatch_size

print("train our model now")
data1 = {"batchsize":[], "loss":[], "error":[]}
for train_times in xrange(1,10):
    data2 = {"batchsize_t":[], "loss_t":[], "error_t":[]}
    for i in range(0, int(num_minibatches_to_train)):
        features=np.float32(training_batch_x[i])
        labels=np.float32(training_batch_y[i])
        # Specify the input variables mapping in the model to actual minibatch data for training
        trainer.train_minibatch({input : features, label : labels})####actual training
        loss = trainer.previous_minibatch_loss_average##################This only look at the last data
        error = trainer.previous_minibatch_evaluation_average###########This only look at the last data
        data2["loss_t"].append(loss)
        data2["error_t"].append(error)
        ###
    loss_all=sum(data2["loss_t"]) / float(len(data2["loss_t"]))
    error_all=sum(data2["error_t"]) / float(len(data2["error_t"]))
    data1["batchsize"].append(train_times)
    data1["loss"].append(loss_all)
    data1["error"].append(error_all)



plt.figure(1)
plt.subplot(211)
plt.plot(data1["batchsize"], data1["loss"], 'b--')
plt.xlabel('Cycle')
plt.ylabel('Loss')
plt.title('Train Cycle run vs. Training loss')
plt.subplot(212)
plt.plot(data1["batchsize"], data1["error"], 'r--')
plt.xlabel('Cycle number')
plt.ylabel('Label Prediction Error')
plt.title('Train Cycle run vs. Label Prediction Error')
plt.show()



features=np.float32(testing_x)
labels=np.float32(testing_y)
out = C.softmax(z)
predicted_label_probs = out.eval({input : features})
print("Label    :", [np.argmax(label) for label in labels])
print("Predicted:", [np.argmax(row) for row in predicted_label_probs])
a=[np.argmax(label) for label in labels]
b=[np.argmax(row) for row in predicted_label_probs]
for iii in xrange(0,len(a)):
    if iii==0:
        temp=((a[iii]==b[iii]).astype(np.float32))
    else:
        temp=temp+((a[iii]==b[iii]).astype(np.float32))

print("The accuracy of our model: ")
print(np.divide(temp, len(labels))*100)
