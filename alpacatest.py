import torch
import os
from tqdm import tqdm
from lxml import etree
import pandas as pd
import os

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch

checkpoint = 'Dogge/alpaca-13b'
print('loading config')

def run_evaluation(checkpoint, gpt_j):
    config = AutoConfig.from_pretrained(checkpoint)
    print('loading model')
    cp_file = checkpoint.replace('/', '_').replace('-', '_')
    cp_file = f'./{cp_file}.pt'
    excel_file = f'{cp_file}.xlsx'

    if not os.path.exists(cp_file):
        model = AutoModelForCausalLM.from_config(config)
        print('saving state dict for use with accelerate')
        torch.save(model.state_dict(), cp_file)

    print('load clean model for use with accelerate')
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()

    print('doing accelerate setup')
    model = load_checkpoint_and_dispatch(
        model, cp_file, device_map='auto', offload_folder='./offload', no_split_module_classes=["GPTJBlock" if gpt_j else "LlamaDecoderLayer"]
    )

    print('loading tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)

    out_df = {
        'file': [],
        'id': [],
        'sub-set': [],
        'prompt': [],
        'sample_solution': [],
        'response': [],
        'feedback': [],
        'predicted_score': [],
        'gt_score': []
    }

    print('iterating')
    for root, dirs, files in tqdm(os.walk('./test/2way/sciEntsBank')):
        for file in files:
            print(file)

            path = os.path.join(root, file)
            if not path.endswith('.xml'):
                continue

            xml_tree = etree.parse(path)
            prompt = ''.join(xml_tree.xpath('/question/questionText')[0].itertext())
            print(prompt)
            sample_solutions = ', '.join(''.join(elem.itertext()) for elem in xml_tree.xpath('/question/referenceAnswers/referenceAnswer'))
            print(sample_solutions)
            print(len(xml_tree.xpath('/question/referenceAnswers/studentAnswers/child::*')))

            for elem in xml_tree.xpath('/question/studentAnswers/studentAnswer'):
                response = ''.join(elem.itertext())
                print(response)
                with torch.no_grad():
                    st = f'''Score the following "Student Answer". Only give a score ("Correct" / "Incorrect"). The "Student Answer" is only correct if it entails the "reference answer", otherwise it is incorrect.

"Task": "{prompt}"

"Reference Answer": "{sample_solutions}"

"Student Answer": "{response}"

Score (which is either "Correct" or "Incorrect"): '''

                    inputs = tokenizer(st, return_tensors="pt")

                    # Generate

                    generate_ids = model.generate(inputs.input_ids, max_length=512)

                    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                    print(out)
                    score = out.replace(st, '')

                    out_df['file'].append(file)
                    out_df['id'].append(elem.get('id'))
                    out_df['prompt'].append(prompt)
                    out_df['sample_solution'].append(sample_solutions)
                    out_df['response'].append(response)
                    out_df['gt_score'].append(elem.get('accuracy'))
                    out_df['predicted_score'].append('incorrect' if ('incorrect' in score.lower() or '0/1' in score) else 'correct')
                    out_df['model_output'].append(out)
                    out_df['sub-set'].append('test-unseen-answers' if 'test-unseen-answers' in path else ('test-unseen-questions' if 'test-unseen-questions' in path else ('test-unseen-domains')))

                    print(out)

                    pd.DataFrame.from_dict(out_df).to_excel(excel_file)

run_evaluation('Dogge/alpaca-13b', False)
run_evaluation('circulus/alpaca-7b', False)
run_evaluation('EleutherAI/gpt-j-6b', True)
                

