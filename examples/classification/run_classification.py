"""Finetuning the library models for sequence classification on GLUE."""

import collections
import copy
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
import logging
import os
from typing import Callable, Dict, Optional

from filelock import FileLock
import numpy as np
from ml_swissknife import utils
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset
from transformers import HfArgumentParser, set_seed

from private_transformers import PrivacyEngine
from .src.common import true_tags
from .src.compiled_args import PrivacyArguments, TrainingArguments, AuxiliaryArguments
from .src.dataset import FewShotDataset
from .src.models import (
    BertForPromptFinetuning, RobertaForPromptFinetuning, AlbertForPromptFinetuning, DistilBertForPromptFinetuning,
    resize_token_type_embeddings
)
from .src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from .src.trainer import Trainer

from transformers import ( 
    WEIGHTS_NAME,
    AdamW,
    AutoConfig, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
) 

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"


ALPHA_INIT=5
PER_PARAMS_ALPHA=1
PER_LAYERS_ALPHA=0

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    static_embedding: str = field(
        default="no"
    )
    static_lm_head: str = field(
        default="no"
    )
    attention_only: str = field(
        default="no"
    )

    randomly_initialize: str = field(
        default="no",
        metadata={"help": "Randomly initialize the model; useful only for ablation studies."}
    )

    def __post_init__(self):
        self.static_embedding = self.static_embedding.lower() in true_tags  # noqa
        self.static_lm_head = self.static_lm_head.lower() in true_tags  # noqa
        self.attention_only = self.attention_only.lower() in true_tags  # noqa
        self.randomly_initialize = self.randomly_initialize.lower() in true_tags  # noqa


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path "
                    "is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when "
                    "prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: tuple = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )

    inference_time_demo: bool = field(
        default=False,
        metadata={"help": "Do not use demonstrations during inference time; "
                          "the original paper attaches to each test example a few training examples as demo -- "
                          "apparently this breaks privacy. We turn this off by default here."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    evaluate_after_training: bool = field(
        default=True, metadata={"help": "Always run evaluation after training ends."}
    )

    def __post_init__(self):
        super(DynamicTrainingArguments, self).__post_init__()


def main():
    parser = HfArgumentParser(
        (ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments, PrivacyArguments, AuxiliaryArguments)
    )
    model_args, data_args, training_args, privacy_args, auxiliary_args = parser.parse_args_into_dataclasses()

    if not os.path.exists(training_args.output_dir):
        print(f"output_dir doesn't exists, mkdir now: {training_args.output_dir}")
        os.makedirs(training_args.output_dir)

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # TODO: Hacky mapping creation. Refactor this in the future.
    #  Currently gets replace if mapping_id and mapping_path is set.
    if data_args.task_name == "sst-2":
        data_args.mapping = "{'0':'terrible','1':'great'}"
    elif data_args.task_name == "mnli":
        data_args.mapping = "{'contradiction': 'no', 'entailment': 'yes', 'neutral': 'maybe'}"
    elif data_args.task_name == "qnli":
        data_args.mapping = "{'not_entailment': 'no', 'entailment': 'yes'}"
    elif data_args.task_name == "qqp":
        data_args.mapping = "{'1': 'yes', '0': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
    else:
        raise ValueError(f"Unknown task: {data_args.task_name}")

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    if model_args.few_shot_type == 'finetune':
        model_fn = AutoModelForSequenceClassification

    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir, use_fast=False
    )
    print(f' | tokenizer: {tokenizer}, size: {len(tokenizer)} \n\n\n')

    # Get our special datasets.
    if model_args.few_shot_type == "finetune":
        assert data_args.num_sample == 1
        train_dataset = GlueDataset(data_args, tokenizer, mode="train")
        eval_dataset = (
            GlueDataset(data_args, tokenizer, mode="dev")
            if training_args.do_eval else None
        )
        test_dataset = (
            GlueDataset(data_args, tokenizer, mode="test")
            if training_args.do_predict or training_args.evaluate_test_split
            else None
        )

        if eval_dataset is not None:
            eval_dataset.num_sample = 1
        if test_dataset is not None:
            test_dataset.num_sample = 1

    print(f" *** dataset sizes: ")
    for _tag, _ds in zip(("train", "valid", "test"), (train_dataset, eval_dataset, test_dataset)):
        if _ds is not None:
            print(f'{_tag}: {len(_ds)}')
    print(f" ***")

    set_seed(training_args.seed)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(" | model type: ")
    print(type(model))


    # ## CHANGED LOCATION, SEE BELOW   
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    ''''''
    model.to(training_args.device)
  
    bert_params = {}
    finetune_params = []
    alpha_params = []
    no_decay = ['bias', 'LayerNorm.weight']
    for n,p in model.named_parameters():
        p0 = torch.zeros_like(p.data).copy_(p) #original BERT
        p1 = torch.zeros_like(p.data) #params to be fine-tuned
        p1.requires_grad = True

        p1.grad = torch.zeros_like(p.data)
        alpha = torch.zeros_like(p.data) + ALPHA_INIT
        alpha.requires_grad = True
        alpha.grad = torch.zeros_like(p.data)

        name = n 

        bert_params[name] = [p0, p1, alpha]
        finetune_params.append(bert_params[name][1])
        alpha_params.append(bert_params[name][2])
    model_device = list(model.named_parameters())[0][1].device

    if PER_PARAMS_ALPHA == 1:
        per_params_alpha_dict = {}
        for n, p in model.named_parameters(): 
            alpha = torch.zeros((1)).to(model_device) + ALPHA_INIT
            alpha.requires_grad=True
            alpha.grad = torch.zeros_like(alpha)

            name = n

            per_params_alpha_dict[name] = alpha
            alpha_params.append(alpha)
    
    optimizer_grouped_parameters = [
        {
            "params": [p[1] for n, p in bert_params.items() if not any(nd in n for nd in no_decay) and p[1].requires_grad is True],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p[1] for n, p in bert_params.items() if any(nd in n for nd in no_decay) and p[1].requires_grad is True], "weight_decay": 0.0},

        {'params': alpha_params, 'lr': 0.1, 'eps': training_args.adam_epsilon, 'weight_decay': 0.0, 'correct_bias': True},
    ]

    '''''' 

    print('===========optimizers done!!!!===============' * 2)
    
    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        model_args=model_args,
        privacy_args=privacy_args,
        auxiliary_args=auxiliary_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    '''''' 
    optimizer = trainer.optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    ''''''    
    # adding additional parameters to trainer
    trainer.bert_params = bert_params
    trainer.finetune_params = finetune_params
    trainer.alpha_params = alpha_params
    trainer.per_params_alpha_dict = per_params_alpha_dict
    trainer.l0_coef = auxiliary_args.l0_coef
    trainer.sparsity_pen_num = auxiliary_args.sparsity_pen_num


    training_setup = trainer.get_training_setup()
    t_total = training_setup["t_total"]
  
    ''''''
    # check that the optimizer is the one we created
    assert trainer.optimizer == optimizer
    # check that the optimizer has the correct number of parameters
    print(trainer.optimizer)

    trainer.create_optimizer_and_scheduler(num_training_steps=t_total)
    ''''''

    if privacy_args.non_private:
        privacy_args.noise_multiplier = 0.
        privacy_args.per_example_max_grad_norm = None

    else:
        total_train_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
        print('In classifiction.py, initializing privacy engine')
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=total_train_batch_size,
            sample_size=len(train_dataset),
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            accounting_mode=privacy_args.accounting_mode,
            clipping_mode=privacy_args.clipping_mode,
            skip_checks=True,
        )
        privacy_engine.bert_params = bert_params
        privacy_engine.finetune_params = finetune_params
        privacy_engine.alpha_params = alpha_params
        privacy_engine.per_params_alpha_dict = per_params_alpha_dict
    
        # Originally, it could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        print('privacy_args: ')
        print(json.dumps(privacy_args.__dict__, indent=4))
        privacy_engine.attach(optimizer)
        print(privacy_engine)

    # Training
    if training_args.do_train:
        # Write argparse.
        utils.jdump(
            {**training_args.__dict__, **model_args.__dict__, **data_args.__dict__, **privacy_args.__dict__},
            os.path.join(training_args.output_dir, 'argparse.json'),
            default=lambda x: str(x),
        )
        print(data_args.mapping)
        print(data_args.template)

        # Don't reload.
        trainer.train(model_path=None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))

    if training_args.do_eval or training_args.do_predict:
        # Reload the best checkpoint (for eval or predict).
        logger.info("*** Loading best checkpoint ***")
        model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    # Evaluation
    final_result = {'time': str(datetime.today())}

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = []
        eval_task_names = []
        eval_splits = []
        for split, dataset in zip(('dev', 'test'), (eval_dataset, test_dataset)):
            if split == "test" and not training_args.evaluate_test_split:
                continue

            eval_datasets.append(dataset)
            eval_task_names.append(data_args.task_name)
            eval_splits.append(split)

            # --- lxuechen: This block depends on `split`.
            if data_args.task_name == "mnli":
                mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
                eval_task_names.append(mnli_mm_data_args.task_name)
                eval_splits.append(split)
                if model_args.few_shot_type == "finetune":
                    mnli_mm_dataset = GlueDataset(mnli_mm_data_args, tokenizer, mode=split)
                    mnli_mm_dataset.num_sample = 1
                    eval_datasets.append(mnli_mm_dataset)
                else:
                    eval_datasets.append(
                        FewShotDataset(
                            mnli_mm_data_args, tokenizer=tokenizer, mode=split,
                            use_demo=('demo' in model_args.few_shot_type),
                        )
                    )
            # ---

        results_json = collections.defaultdict(dict)
        for eval_dataset, eval_task_name, eval_split in zip(eval_datasets, eval_task_names, eval_splits):
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            # --- lxuechen: My evaluation procedure.
            if eval_result is not None:
                if not privacy_args.non_private:
                    privacy_spent = privacy_engine.get_privacy_spent(accounting_mode="all", lenient=True)
                    to_record_dict = {**eval_result, **privacy_spent}
                else:
                    to_record_dict = eval_result

                if training_args.evaluate_test_split:
                    results_json[eval_split][eval_task_name] = to_record_dict
                else:
                    results_json[eval_task_name] = to_record_dict
            # ---

        output_path = os.path.join(training_args.output_dir, "final_results.json")
        utils.jdump(results_json, output_path)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")

        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            if model_args.few_shot_type == "finetune":
                mnli_mm_dataset = GlueDataset(mnli_mm_data_args, tokenizer, mode="test")
                mnli_mm_dataset.num_sample = 1
                test_datasets.append(mnli_mm_dataset)
            else:
                test_datasets.append(
                    FewShotDataset(
                        mnli_mm_data_args,
                        tokenizer=tokenizer, mode="test", use_demo=('demo' in model_args.few_shot_type)
                    )
                )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir,
                                         "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id,
                                                               training_args.array_id)), logits)

            test_results.update(test_result)

    with FileLock('log.lock'):
        with open('log', 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')

    return eval_results


if __name__ == "__main__":
    main()
