Hello and welcome to my project.

Using both Tensorflow and Sklearn I built a program that allows the user to train a model using the messages in a WhatsApp .txt exported file in order to then be able to predict the author of a completely new and invented phrase. 
The program first parses the texts and authors and saves them as a .pkl file. After that the authors are then encoded andf the texts using Tensorflow's TextVectorization are vectorized in order for the network to be able to interpret and be trained. 
Both the classifier model and encoder are then saved to be used later: the model as a .keras and the encoder as a .pkl to avoid having to retrain the model on the same chat file. The terminal usage for the program is as follows: 

'Usage: main.py (search/predict/train) (word/phrase to search or predict, or filename for training)' 

and is important to note that the chat .txt file should be located inside the same folder as the main.py file, 
therefore it is recomended to create a folder for both files as the other files created by the program will also be stored there. In the case that multiple different models want to be trained using different chats, 
I suggest to create a folder for each and run the program on each folder as there is no multiple model capability implemented. Finally, it is important to note that this program was based on WhatsApp exported .txt files so different app chats or formats will
likely not work.

Pedro-Oub 2025
