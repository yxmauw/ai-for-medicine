import numpy as np
import pandas as pd
import tensorflow as tf

### ex1 & ex2
def test_sentences_test_case():
    test_sentences = ["Diffuse Reticular Pattern, which can be seen with an atypical infection or chronic fibrotic change.", 
                      "no Focal Consolidation."]
    
    return test_sentences

### ex3
def prepare_bert_input_test_case():
    passage = "My name is Bob."
    question = "What is my name?"
    
    return passage, question

### ex4
def get_span_from_scores_test_case():
    start_scores_1 = tf.convert_to_tensor([-1, 2, 0.4, -0.3, 0, 8, 10, 12], dtype=float)
    end_scores_1 = tf.convert_to_tensor([5, 1, 1, 3, 4, 10, 10, 10], dtype=float)
    input_mask_1 = [1, 1, 1, 1, 1, 0, 0, 0]
    
    start_scores_2 = tf.convert_to_tensor([0, 2, -1, 0.4, -0.3, 0, 8, 10, 12], dtype=float)
    end_scores_2 = tf.convert_to_tensor([0, 5, 1, 1, 3, 4, 10, 10, 10], dtype=float)
    input_mask_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0 ]
    
    return start_scores_1, end_scores_1, input_mask_1, start_scores_2, end_scores_2, input_mask_2

### ex5
def construct_answer_test_case():
    tmp_tokens_1 = [' ## hello', 'how ', 'are ', 'you?      ']
    tmp_tokens_2 = ['@',' ## hello', 'how ', 'are ', 'you?      ']
    
    return tmp_tokens_1, tmp_tokens_2