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

### Evaluation Tasks
10.1, 10.2, 10.3, 10.4

#### Deliverables:
Report results for two model configurations: with dropout and without dropout
1. Code for implementing your model including vocabulary creation
2. Final model checkpoints
3. Scores of your models for each evaluation task
4. Time taken to run 1 epoch for each task. How did you improve this running time?


### Bonus
Compare your model on evaluation tasks with [Universal Sentence Encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1) 