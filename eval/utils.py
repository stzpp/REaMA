import torch
import tqdm
import json
import time
import asyncio
import os
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
#from open_instruct.finetune import encode_with_prompt_completion_format
from eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)
    
    
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")
        #print('pad_token_id', tokenizer.pad_token, tokenizer.pad_token_id)
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            #print('batch_input_ids', batch_input_ids)
            
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
            #print('batch_outputs', batch_outputs)
            #print('generation_kwargs', generation_kwargs)
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            #print('batch_prompts', batch_prompts)
            #print('batch_prompts-skip', tokenizer.batch_decode(batch_input_ids, skip_special_tokens=False))
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()
        #print('batch_input_ids', batch_input_ids[0])
        batch_logits = model(batch_input_ids, attention_mask).logits[:, -1, :]
        top10_indices = batch_logits.topk(10).indices
        print('prompts:')
        print(batch_prompts[0])#.replace('\n\n', '\n'))
        print('ABCD ids:')
        print(candidate_token_ids)
        print('generated top-10 ids:')
        print(top10_indices)
        print('generated top-10:')
        print(tokenizer.batch_decode(top10_indices))
        # return
        if candidate_token_ids is not None:
            print('logits.mean:')
            print(batch_logits.mean())
            batch_logits = batch_logits[:, candidate_token_ids]
        print('logits of ABCD:')
        print(batch_logits)
        batch_probs = torch.softmax(batch_logits, dim=-1)
        print('softmax probs of ABCD:')
        print(batch_probs)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores



def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        load_in_8bit=False, 
        load_in_half=False,
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
    ):
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    if "Qwen" in model_name_or_path:


        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )
        print("Loading model ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            pad_token_id=tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, pad_token_id=tokenizer.pad_token_id)
        
    elif "chatglm" in model_name_or_path: 

        #from modelscope import AutoTokenizer, AutoModel, snapshot_download
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()  
    elif "Baichuan" in model_name_or_path: 

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
    elif "deepseek" in model_name_or_path:

        config_name = "deepseek-ai/deepseek-llm-67b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #print('tokenizer.eos_token_id', tokenizer.eos_token_id)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        #print('tokenizer.eos_token_id', tokenizer.eos_token_id)
        #print('tokenizer.pad_token_id', tokenizer.pad_token_id)
        
    else:


        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
        # set padding side to left for batch generation

        if gptq_model:
            from auto_gptq import AutoGPTQForCausalLM
            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path, device="cuda:0", use_triton=True
            )
            model = model_wrapper.model  
        elif load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                device_map=device_map, 
                load_in_8bit=True
            )
        else:
            if device_map:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                if torch.cuda.is_available():
                    model = model.cuda()
            if load_in_half:
                model = model.half()

    #from transformers import GenerationConfig      
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    #print('tokenizer.pad_token', tokenizer.pad_token)
    #print('tokenizer.eos_token', tokenizer.eos_token)
    #model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    #print('tokenizer.pad_token', tokenizer.pad_token)
    #print('tokenizer.eos_token', tokenizer.eos_token)
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token_id is None:
        ##tokenizer.pad_token = "<unk>"
        #tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        #model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        #model.generation_config.pad_token_id = model.generation_config.eos_token_id

    #print('tokenizer.eos_token_id', tokenizer.eos_token_id)
    #print('tokenizer.pad_token_id', tokenizer.pad_token_id)
    model.eval()
    return model, tokenizer



def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                    messages_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["message"]["content"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 

def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                    prompt_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["text"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 