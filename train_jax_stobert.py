import argparse
import argparse
import logging
import math
import os
import random


from pathlib import Path
import datasets
import torch
from tqdm.auto import tqdm
import numpy as np
import sys
import time

from accelerate import Accelerator
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    # AdamW,
    # AutoConfig,
    # AutoModelForMaskedLM,
    # AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed
)

from scipy.stats import entropy
import torch.distributions as D
import evaluate

from data import get_nli_dataset
#from models.modeling_bert import BertForSequenceClassification
from models.modeling_flax_stobert import (
        FlaxStoBertForSequenceClassification, 
        FlaxStoSequenceClassifierOutput,
)
#from models.modeling_outputs import StoSequenceClassifierOutput
from models.config import StoBertConfig
from data import get_nli_dataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Bert model on NLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default='mnli-m-chaosnli',
        choices=[
            "mnli-mm",
            "mnli-m",
            "mnli-m-small", # half of the mnli-m dataset
            "mnli-m-chaosnli",
            "mnli-m-small-chaosnli",  # half of the mnli-m dataset
            "snli",
            "mnli-snli",
            "snli-mnli-m",
            "snli-mnli-mm",
            "snli-sick",
            "mnli-sick",
        ]
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to directory containing NLI datasets",
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        default=3,
        help="Number of classes for classification. For NLI, there are 3 classes.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Max. train steps per epoch",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.",
                        )
    parser.add_argument(
        "--logging_freq",
        type=int,
        default=10,
        help="logging frequency during training.",
    )

    args = parser.parse_args()

    return args


def count_parameters(model, logger):
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def get_vi_weight(epoch, kl_min, kl_max, last_iter):
    value = (kl_max-kl_min)/last_iter
    return min(kl_max, kl_min + epoch*value)


def evaluate_stochastic(model, dataloader, num_samples, top_n=1):
    tnll = 0
    y_prob = []
    y_true = []
    y_prob_all = []
    acc_metric = evaluate.load("accuracy")
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            x = batch['input_ids']
            new_shape = (x.shape[0], num_samples*model.n_components, model.num_labels)
            prob = torch.cat([model(**batch, n_samples=num_samples, indices=torch.full((x.size(0)*num_samples,), idx)).logits for idx in range(model.n_components)], dim=1)
            prob = prob.reshape(new_shape)
            y_target = batch['labels'].unsqueeze(1).expand(-1, num_samples*model.n_components)
            # print('prob:', prob.shape)
            # print('y_target:', y_target.shape)
            bnll = D.Categorical(logits=prob).log_prob(y_target)
            bnll = torch.logsumexp(bnll, dim=1) - torch.log(torch.tensor(num_samples*model.n_components, dtype=torch.float32, device=bnll.device))
            tnll -= bnll.sum().item()
            vote = prob.exp().mean(dim=1)
            top_pred = torch.topk(vote, k=top_n, dim=1, largest=True, sorted=True)[1]
            y_prob_all.append(prob.exp().cpu().numpy())
            y_prob.append(vote.cpu().numpy())
            # y_true.append(by.cpu().numpy())
            # print('top_pred:', top_pred.shape)
            # print('labels:', batch['labels'].shape)
            acc_metric.add_batch(predictions=top_pred, references=batch['labels'])
    acc_res = acc_metric.compute()
    y_prob = np.concatenate(y_prob, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    total_entropy = entropy(y_prob, axis=1)
    aleatoric = entropy(y_prob_all, axis=-1).mean(axis=-1)
    epistemic = total_entropy - aleatoric
    result = {
        'accuracy': acc_res,
        'predictive_entropy': {
            'total': (float(total_entropy.mean()), float(total_entropy.std())),
            'aleatoric': (float(aleatoric.mean()), float(aleatoric.std())),
            'epistemic': (float(epistemic.mean()), float(epistemic.std()))
        }
    }
    return result


def main():

    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    #accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # If passed along, set the training seed now.
    #if args.seed is not None:
    #    set_seed(args.seed)

    # Tokenizer
    config = StoBertConfig()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

        # Model
    #model = StoBertForSequenceClassification(config).from_pretrained(args.model_name_or_path, config=config, num_labels=3)
    config.num_labels = args.num_labels
    model = FlaxStoBertForSequenceClassification(config).from_pretrained(
        args.model_name_or_path, config=config
    )
    
    import sys
    sys.exit(1)

    
    # Data loaders
    config.dataset = args.dataset
    train_dataloader, dev_dataloader, test_dataloader = get_nli_dataset(config, tokenizer, args.data_path)
    print('train_dataloader:', len(train_dataloader))
    print('dev_dataloader:', len(dev_dataloader))
    print('test_dataloader:', len(test_dataloader))

# Optimizer: separate deterministic & stochastic params
    det_params = config.det_params
    sto_params = config.sto_params

    detp = []
    stop = []
    for name, param in model.named_parameters():
        if 'posterior' in name or 'prior' in name:
            stop.append(param)
        else:
            detp.append(param)

    config.sto_params['lr'] = args.learning_rate
    config.det_params['lr'] = args.learning_rate
    optimizer_grouped_parameters = [
        {
             'params': detp,
             **det_params
         }, {
             'params': stop,
             **sto_params
         },
    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Count parameters to train
    count_parameters(model, logger)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    n_batch = len(train_dataloader)
    print("--- Training start ---")
    model.train()

    for epoch in range(args.num_train_epochs):
        print('--- epoch', epoch, '---')
        beta = get_vi_weight(epoch=epoch,
                             kl_min=config.kl_weight['kl_min'],
                             kl_max=config.kl_weight['kl_max'],
                             last_iter=config.kl_weight['last_iter'])
        print('beta:', beta)
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, n_samples=config.num_train_samples)
            loss = outputs.loss + beta*(outputs.kl - config.gamma * outputs.entropy)/(n_batch*batch['input_ids'].size(0))
            # loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            # if step >= args.max_train_steps:
            #     break

        # metric = evaluate.load("accuracy")
        # model.eval()
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         outputs = model(**batch, n_samples=config.num_test_sample)
        #     logits = outputs.logits
        #     predictions = torch.argmax(logits, dim=-1)
        #     metric.add_batch(predictions=predictions, references=batch["labels"])
        # res = metric.compute()
        # print('result:', res)
    print("--- Training end ---")

    print('--- Test StoBERT and compute uncertainty measures ---')
    # Evaluate the modelgradient_accumulation_steps
    result = evaluate_stochastic(model=model, dataloader=test_dataloader, num_samples=config.num_test_samples)
    print(result)

    print("--- Save model ---")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
