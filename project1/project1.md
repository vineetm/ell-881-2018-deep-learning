**Paper Title**: "Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning"

* Paper [Link](https://openreview.net/forum?id=B18WgG-CZ&noteId=B18WgG-CZ)

* [Github Repo](https://github.com/Maluuba/gensen)

* Code is available in pytorch. We don't want you to port the code from pytorch to tensorflow!

* Implement your code in tf.eager, use tf.keras.Model to build components of your model, as shown in RNN notebook

### Project Details

#### Tasks to be implemented:
1. NLI
2. Constituency Parsing
3. NMT on En-De

#### Training Flow
1. Train Model for 1 epoch on Task 1, save checkpoints which give best results on corresponding dev file
2. Start from saved checkpoints for Task 1. Train with Task 2, save checkpoints
3. Repeat for next task


#### Speeding up data pipeline
See tips for speeding up data pipeline [here](https://cs230-stanford.github.io/tensorflow-input-data.html) Specifically use multiple threads and prefetching!

#### Model details
1. **RNN Cell**: Use only unidirectional GRU
2. Restrict SRC Vocab to 30,000 words. Pick most frequently occuring words across all datasets 
3. Use word embedding dimension as 256 and GRU cell dimension as 512.
4. Compare your models with and without dropout (0.3)
5. Use Adam optimizer with learning rate of 0.0002
6. Use batch size of 32
7. Rest of the parameters should be used as mentioned in  

#### Evaluation Tasks
10.1, 10.2, 10.3, 10.4


#### Guidelines
1. Do not directly run your experiments on GPU. Verify if you model works by first trying to overfit it on a small portion of training data on your local system!
2. Save model checkpoints every hour or so. Checkpoints will allow you to resume your work in case your training job gets killed!
3. Use a fixed random seed for your experiments.

#### What you need to submit
Prepare a single zip which contains all the following:
1. Code for data pipeline for your model (tf.data). Add comments for your code!
2. Code for your model. Add comments for your code! 
3. Source Vocabulary file
4. Training logs for your 2 final models: with dropout, without dropout
    * When running your training, don't use prints but tf.logging
    * It is set as follows, in the beginning of your main program:    
    ```python
       logging = tf.logging
       logging.set_verbosity(logging.INFO)
 
       def log_msg(msg):
           logging.info(f'{time.ctime()}: {msg}')
    ```
    
    * Now, you can print as follows:
    ```
    log_msg(f'Epoch: {epoch_num} Step: {step_num} ppl improved: {ppl: 0.4f}')   
    ```
5. 5 nearest neighbors for each word in your source vocabulary, computed using Embedding. Note, we want **words** and not their integer indexes! File should look as follows:
    ```bash
       word1, neighbor1, neighbor2, neighbor3, neighbor4, neighbor5
       word2, ....
       ....
       word30000, ....
    ```
    *Hint* Use: [sklearn cosine similarity](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) to find NNs 
6. Time taken to run 1 epoch for each task on CPU and GPU. How did you make this faster?
7. Scores for all the evaluation tasks 
 

### Bonus
Compare your model on evaluation tasks with [Universal Sentence Encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1) 