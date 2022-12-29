import numpy as np
import pandas as pd
import seaborn as sns
from test_utils import *
from test_case import *
import tensorflow as tf

### ex1
def get_labels_test(target):
    
    test_sentences = test_sentences_test_case()
    
    print("Test Case:\n")
    print("Test Sentences:\n")
    for s in test_sentences:
        print(s)
    print("\n")
    
    print("Retrieved Labels:\n")
    retrieved_labels = target(test_sentences)
    for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print("\n")
    
    expected_output = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': True, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    


##############################################        
### ex2
def get_labels_negative_aware_test(target):
    
    test_sentences = test_sentences_test_case()
    
    print("Test Case:\n")
    print("Test Sentences:\n")
    for s in test_sentences:
        print(s)
    print("\n")
    
    print("Retrieved Labels:\n")
    retrieved_labels = target(test_sentences)
    for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]):
        print("{} : {}".format(key, value))
    print("\n")
    
    expected_output = expected_labels = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': False, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [test_sentences],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
    



##############################################        
### ex3
def prepare_bert_input_test(target, tokenizer):
    
    passage, question = prepare_bert_input_test_case()
    
    print("Test Case:\n")
    print("Passage: ", passage)
    print("Question: ", question, "\n")
    
    max_seq_length = 20
    input_ids, input_mask, tokens = target(question, passage, tokenizer, max_seq_length)
    print("Tokens:")
    print(tokens)
    print("\nCorresponding input IDs:")
    print(input_ids)
    print("\nMask:")
    print(input_mask, "\n")
    
    def expected_output_values():
        input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        tokens = ['[CLS]', 'What', 'is', 'my', 'name', '?', '[SEP]', 'My', 'name', 'is', 'Bob', '.']
        input_ids = [101, 1327, 1110, 1139, 1271, 136, 102, 1422, 1271, 1110, 3162, 119, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids), 0)
        
        return input_ids, input_mask, tokens
    
    expected_output = expected_output_values()
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Data-type mismatch."
        },
        {
            "name": "shape_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [question, passage, tokenizer, max_seq_length],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    multiple_test(test_cases, target)
        

##############################################    
### ex4
def get_span_from_scores_test(target):
    
    start_scores_1, end_scores_1, input_mask_1, start_scores_2, end_scores_2, input_mask_2 = get_span_from_scores_test_case()
    
    verbose = True
    
    print("Test Case 1: \n")
    start_1, end_1 = target(start_scores_1, end_scores_1, input_mask_1, verbose)
    print("Expected: (1, 4)")
    print("Returned: ({}, {})".format(start_1, end_1))
    expected_output_1 = (1, 4)
    
    print("\nTest Case 2: \n")
    start_2, end_2 = target(start_scores_2, end_scores_2, input_mask_2, verbose)
    print("Expected: (1, 1)")
    print("Returned: ({}, {})".format(start_2, end_2))
    print("\n")
    expected_output_2 = (1, 1)
    
    verbose = False
    test_cases = [
        {
            "name":"datatype_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [start_scores_1, end_scores_1, input_mask_1, verbose],
            "expected": expected_output_1,
            "error": "Wrong output for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [start_scores_2, end_scores_2, input_mask_2, verbose],
            "expected": expected_output_2,
            "error": "Wrong output for Test Case 2. Please check how you set the range of your for loops."
        }
    ]
    
    multiple_test(test_cases, target)
    
    
##############################################    
### ex5
def construct_answer_test(target):
    
    tmp_tokens_1, tmp_tokens_2 = construct_answer_test_case()
    
    print("Test Case: \n")
    print("Originals:\n")
    print(tmp_tokens_1)
    print(tmp_tokens_2)
    
    tmp_out_string_1 = target(tmp_tokens_1)
    tmp_out_string_2 = target(tmp_tokens_2)
    
    expected_output_1 = "hello how  are  you?"
    expected_output_2 = "@hellohowareyou?"
    
    print("\nProcessed Strings:\n")
    print(f"tmp_out_string_1: {tmp_out_string_1}, length {len(tmp_out_string_1)}")
    print(f"tmp_out_string_2: {tmp_out_string_2}, length {len(tmp_out_string_2)}")
    print("\n")
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Data-type mismatch for Test Case 1."
        },
        {
            "name": "shape_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Wrong shape for Test Case 1."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_tokens_1],
            "expected": expected_output_1,
            "error": "Wrong output for the first string."
        },
        {
            "name": "equation_output_check",
            "input": [tmp_tokens_2],
            "expected": expected_output_2,
            "error": "Wrong output for the second string."
        }
    ]
    
    multiple_test(test_cases, target)