
import torch
import os
from tqdm import tqdm
from lxml import etree
import pandas as pd
import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import random

device = torch.device('cpu')

def load_data_xml_file_semeval(file):
    id = file[file.rfind('/') + 1 : ]
    xml_tree = etree.parse(file)
    prompt = ''.join(xml_tree.xpath('/question/questionText')[0].itertext())
    sample_solutions = '; '.join(''.join(elem.itertext()) for elem in xml_tree.xpath('/question/referenceAnswers/referenceAnswer'))

    elems = xml_tree.xpath('/question/studentAnswers/studentAnswer')
    ddf = {
        "id": [id for elem in elems],
        "question": [prompt for elem in elems],
        "reference_answer": [sample_solutions for elem in elems],
        "student_answer": [''.join(elem.itertext()) for elem in elems],
        "score": [elem.get('accuracy') for elem in elems]
    }
    return pd.DataFrame.from_dict(ddf)

def load_complete_dataset_semeval(path):    
    dfs = [
        load_data_xml_file_semeval(os.path.join(root, file))
        for root, dirs, files in os.walk(path)
        for file in files
        if file.endswith('.xml')
    ]

    return pd.concat(dfs, axis=0, ignore_index=True)

def load_complete_dataset_saf(path):
    dataset = pd.read_parquet(path, engine="auto")
    
    ddf = {
        "id": dataset["id"].values,
        "question": dataset["question"].values,
        "reference_answer": dataset["reference_answer"].values,
        "student_answer": dataset["provided_answer"].values,
        "score": dataset["verification_feedback"].values 
    }

    return pd.DataFrame.from_dict(ddf)

def create_prompt(base_prompt, train_data, n_shot, test_data, index, chat_mode):
    question_format = '''Question: "{prompt}"\nReference Answer(s): "{sample_solution}"\nStudent Answer: "{response}"\nScore: {score}'''

    n_shot_examples = []
    for n in range(n_shot):
        idx = random.randint(0, len(train_data) - 1)
        n_shot_examples.append(
            question_format
            .replace('{prompt}', train_data['question'].values[idx])
            .replace('{sample_solution}', train_data['reference_answer'].values[idx])
            .replace('{response}', train_data['student_answer'].values[idx])
            .replace('{score}', train_data['score'].values[idx].capitalize())
        )
    n_shot_examples.append(
        question_format
        .replace('{prompt}', test_data['question'].values[index])
        .replace('{sample_solution}', test_data['reference_answer'].values[index])
        .replace('{response}', test_data['student_answer'].values[index])
        .replace('{score}', "")
    )
    n_shot_str = '\n\n'.join(n_shot_examples)
    return f'<s>[INST] {base_prompt}{n_shot_str} [/INST]' if chat_mode else f'{base_prompt}{n_shot_str}'

def create_prompt_set(base_prompt, train_data, n_shot, test_data, chat_mode):
    return [
        create_prompt(base_prompt, train_data, n_shot, test_data, i, chat_mode) for i in range(len(test_data))
    ]

def run_experiment(model, tokenizer, base_prompt, train_data, n_shot, test_data, chat_mode):
    prompt_set = create_prompt_set(
        base_prompt,
        train_data,
        n_shot,
        test_data,
        chat_mode
    )

    with torch.no_grad():
        outputs = [
            tokenizer.batch_decode(
                model.generate(
                    **tokenizer(
                        prompt_set[i],
                        padding=True,
                        truncation=True, 
                        return_tensors="pt"
                    ).to(device),
                    max_new_tokens=1000,
                    do_sample=True
                ),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            for i
            in tqdm(range(len(test_data)))
        ]

    ddf = {
        "n_shot": [str(n_shot) for i in range(len(test_data))],
        "question": [test_data['question'].values[i] for i in range(len(test_data))],
        "referense_answer": [test_data['reference_answer'].values[i] for i in range(len(test_data))],
        "student_answer": [test_data['student_answer'].values[i] for i in range(len(test_data))],
        "base_prompt": [base_prompt for i in range(len(test_data))],
        "prompt": prompt_set,
        "outputs": outputs,
        "predicted_scores": [o[ : o.index('.')] for o in outputs],
        "ground_truth_scores": [test_data['score'].values[i] for i in range(len(test_data))]
    }

    return pd.DataFrame.from_dict(ddf)

# SemEval

prompt_scoring_2way = 'Your task is to score the "Student Answer" to the "Question". Give a score which is either "Correct" or "Incorrect", depending on if the "Student Answer" matches the "Reference Answer". Explain your reasoning step by step.\n\n'
prompt_scoring_3way = 'Your task is to score the "Student Answer" to the "Question". Give a score which is either "Correct", "Incorrect" or "Contradictory", depending on if the "Student Answer" matches the "Reference Answer". Explain your reasoning step by step.\n\n'

prompt_scoring_teacher_2way = 'You are a teacher whose task is to score the "Student Answer" to the "Question". Give a score which is either "Correct" or "Incorrect", depending on if the "Student Answer" matches the "Reference Answer". Explain your reasoning step by step.\n\n'
prompt_scoring_teacher_3way = 'Your are a teacher whose task is to score the "Student Answer" to the "Question". Give a score which is either "Correct", "Incorrect" or "Contradictory", depending on if the "Student Answer" matches the "Reference Answer". Explain your reasoning step by step.\n\n'

prompt_entailment_2way = 'Your task is to score the "Student Answer" to the "Question". Give a score which is either "Correct" or "Incorrect", depending on if a human reading of the "Reference Answer" would be justified in inferring the proposition expressed by the "Student Answer" from the proposition expressed by the "Reference Answer". Explain your reasoning step by step.\n\n'
prompt_entailment_3way = 'Your task is to score the "Student Answer" to the "Question". Give a score which is either "Correct", "Incorrect" or "Contradictory", depending on if a human reading of the "Reference Answer" would be justified in inferring the proposition expressed by the "Student Answer" from the proposition expressed by the "Reference Answer". Explain your reasoning step by step.\n\n'

prompt_entailment_teacher_2way = 'You are a teacher whose task is to score the "Student Answer" to the "Question". Give a score which is either "Correct" or "Incorrect", depending on if a human reading of the "Reference Answer" would be justified in inferring the proposition expressed by the "Student Answer" from the proposition expressed by the "Reference Answer". Explain your reasoning step by step.\n\n'
prompt_entailment_teacher_3way = 'You are a teacher whose task is to score the "Student Answer" to the "Question". Give a score which is either "Correct", "Incorrect" or "Contradictory", depending on if a human reading of the "Reference Answer" would be justified in inferring the proposition expressed by the "Student Answer" from the proposition expressed by the "Reference Answer". Explain your reasoning step by step.\n\n'

print('load training sets')
training_2way_beetle = load_complete_dataset_semeval('./SemEval/training/2way/beetle')
training_3way_beetle = load_complete_dataset_semeval('./SemEval/training/3way/beetle')
training_2way_scientsbank = load_complete_dataset_semeval('./SemEval/training/2way/sciEntsBank')
training_3way_scientsbank = load_complete_dataset_semeval('./SemEval/training/3way/sciEntsBank')
training_saf_english = pd.concat((load_complete_dataset_saf('./SAF-english/train.parquet'), load_complete_dataset_saf('./SAF-english/validation.parquet')), axis=0)
training_saf_german_legal = pd.concat((load_complete_dataset_saf('./SAF-german-legal/train.parquet'), load_complete_dataset_saf('./SAF-german-legal/validation.parquet')), axis=0)
training_saf_german_microjob = pd.concat((load_complete_dataset_saf('./SAF-german-microjob/train.parquet'), load_complete_dataset_saf('./SAF-german-microjob/validation.parquet')), axis=0)

print('load test sets')
test_2way_beetle_unseen_answers = load_complete_dataset_semeval('./SemEval/test/2way/beetle/test-unseen-answers')
test_2way_beetle_unseen_questions = load_complete_dataset_semeval('./SemEval/test/2way/beetle/test-unseen-questions')
test_3way_beetle_unseen_answers = load_complete_dataset_semeval('./SemEval/test/3way/beetle/test-unseen-answers')
test_3way_beetle_unseen_questions = load_complete_dataset_semeval('./SemEval/test/3way/beetle/test-unseen-questions')
test_2way_scientsbank_unseen_answers = load_complete_dataset_semeval('./SemEval/test/2way/sciEntsBank/test-unseen-answers')
test_2way_scientsbank_unseen_questions = load_complete_dataset_semeval('./SemEval/test/2way/sciEntsBank/test-unseen-questions')
test_2way_scientsbank_unseen_domains = load_complete_dataset_semeval('./SemEval/test/2way/sciEntsBank/test-unseen-domains')
test_3way_scientsbank_unseen_answers = load_complete_dataset_semeval('./SemEval/test/3way/sciEntsBank/test-unseen-answers')
test_3way_scientsbank_unseen_questions = load_complete_dataset_semeval('./SemEval/test/3way/sciEntsBank/test-unseen-questions')
test_3way_scientsbank_unseen_domains = load_complete_dataset_semeval('./SemEval/test/3way/sciEntsBank/test-unseen-domains')
test_saf_english_unseen_answers = load_complete_dataset_saf('./SAF-english/test_unseen_answers.parquet')
test_saf_english_unseen_questions = load_complete_dataset_saf('./SAF-english/test_unseen_questions.parquet')
test_saf_german_legal_unseen_answers = load_complete_dataset_saf('./SAF-german-legal/test_unseen_answers.parquet')
test_saf_german_legal_unseen_questions = load_complete_dataset_saf('./SAF-german-legal/test_unseen_questions.parquet')
test_saf_german_microjob_unseen_answers = load_complete_dataset_saf('./SAF-german-microjob/test_unseen_answers.parquet')
test_saf_german_microjob_unseen_questions = load_complete_dataset_saf('./SAF-german-microjob/test_unseen_questions.parquet')

# SAF English



# SAF German