import torch
import os
from tqdm import tqdm
from lxml import etree
import pandas as pd
import os

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

st = '''Your task is to score the "Student Answer". Only give a score ("Correct" / "Incorrect"). The "Student Answer" is only correct if it entails the "Reference Answer", otherwise it is incorrect.
Here are some examples of scored "Student Answers":
"Task": "Georgia found one brown mineral and one black mineral. How will she know which one is harder?"
"Reference Answer": "The harder mineral will leave a scratch on the less hard mineral. If the black mineral is harder, the brown mineral will have a scratch."
"Student Answer": "The one with scratches or deeper scratches is weaker and the other rock is harder."
"Score": Correct
"Task": "Carrie wanted to find out which was harder, a penny or a nickel, so she did a scratch test. How would this tell her which is harder?"
"Reference Answer": "The harder coin will scratch the other."
"Student Answer": "Rub them against a crystal."
"Score": Incorrect
"Task": "Denise made a circuit to light a bulb or run a motor off a D-cell battery. She used a special switch. Below is the schematic diagram of her circuit. The switch is inside the dotted box. Why will the bulb light when she moves the switch to the left?"
"Reference Answer": "There is a complete circuit connecting the bulb to the D-cell battery."
"Student Answer": "It will happen because electricity is flowing to the light bulb."
"Score": Correct
"Task": "Maggie wanted to find out if surface area affected temperature change. She had 3 white cake pans. She filled each with 300 milliliters of water. She put one small black plastic disk in pan A, 2 in pan B, and 3 in pan C. Then she put all 3 pans in the sun. When Maggie measured the water temperature in each pan after 15 minutes, which pan do you think held the hottest water? Explain your answer."
"Reference Answer": "C. Black absorbs more heat (energy) than white. Pan C has the most dark surface area so C would heat up the fastest and have the highest temperature."
"Student Answer": "C. Because there are 3 heat sinks on C which will keep the pan warm in the night."
"Score": Incorrect

"Task": "{prompt}"
"Reference Answer": "{sample_solution}"
"Student Answer": "{response}"
"Score": '''

def run_evaluation(checkpoint, gpt_j=False, batch_size=8):
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
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

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

    for root, dirs, files in tqdm(os.walk('./test/2way/sciEntsBank')):
        for file in files:

            path = os.path.join(root, file)
            if not path.endswith('.xml'):
                continue

            xml_tree = etree.parse(path)
            prompt = ''.join(xml_tree.xpath('/question/questionText')[0].itertext())
            sample_solutions = ', '.join(''.join(elem.itertext()) for elem in xml_tree.xpath('/question/referenceAnswers/referenceAnswer'))

            for i, elem in enumerate(xml_tree.xpath('/question/studentAnswers/studentAnswer')):
                response = ''.join(elem.itertext())

                out_df['file'].append(file)
                out_df['id'].append(elem.get('id'))
                out_df['prompt'].append(prompt)
                out_df['sample_solution'].append(sample_solutions)
                out_df['response'].append(response)
                out_df['gt_score'].append(elem.get('accuracy'))
                out_df['predicted_score'].append('-')
                out_df['model_output'].append('-')
                out_df['sub-set'].append('test-unseen-answers' if 'test-unseen-answers' in path else ('test-unseen-questions' if 'test-unseen-questions' in path else ('test-unseen-domains')))

    dataset = pd.DataFrame.from_dict(out_df)
    
    num_batches = len(dataset['file'].values) // batch_size + (1 if (len(dataset['file'].values) % batch_size != 0) else 0)

    for b in range(num_batches):
        min_idx = b * batch_size
        max_idx = b * batch_size + (batch_size if (b < num_batches - 1) else num_batches)

        responses = out_df['response'].values[min_idx:max_idx]
        prompts = out_df['prompt'].values[min_idx:max_idx]
        sample_sols = out_df['sample_solution'].values[min_idx:max_idx]

        prompts = [
            st.format(prompt=prompts[jj], sample_solution=sample_sols[jj], response=responses[jj])
            for jj
            in range(len(responses))
        ]

        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt").to(0)

            generate_ids = model.generate(inputs.input_ids)

            outs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            scores = [out.replace(st, '') for out in outs]
            for jj, score in enumerate(scores):
                dataset['predicted_score'][b * batch_size + jj] = 'incorrect' if ('incorrect' in score.lower() or '0/1' in score) else 'correct'
            for jj, out in enumerate(outs):
                dataset['model_output'][b * batch_size + jj] = out
            
            dataset.to_excel(excel_file)

run_evaluation('Dogge/alpaca-13b', False, 6)
run_evaluation('EleutherAI/gpt-j-6b', True, 12)
run_evaluation('circulus/alpaca-7b', False, 12)
