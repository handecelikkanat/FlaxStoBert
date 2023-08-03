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

import jax
import flax
import optax

from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from flax.training import train_state

from typing import Callable

#from models.modeling_bert import BertForSequenceClassification
from models.modeling_flax_stobert import (
        FlaxStoBertForSequenceClassification, 
        FlaxStoSequenceClassifierOutput,
)
#from models_nli.modeling_outputs import StoSequenceClassifierOutput
from models.config import StoBertConfig
from data_nli import get_nli_datasets

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



#def loss_function(logits, labels):
#    if is_regression:
#        return jnp.mean((logits[..., 0] - labels) ** 2)
#
#    xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
#    return jnp.mean(xentropy)

def eval_function(logits):
    return logits[..., 0] if is_regression else logits.argmax(-1)


class TrainState(train_state.TrainState):
    logits_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)


def train_step(state, batch, train_step_rng):
    targets = batch.pop("labels")
    dropout_rng, categorical_rng, new_train_step_rng = jax.random.split(train_step_rng, 3)

    def loss_function(params):

    # TODO: out apply function returns FlaxStoSequenceClassifierOutput
        output = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        logits = output.logits

        loss = state.loss_function(logits, targets)

        #for calculating the kl loss and entropy, now we use categorical_rng

        #Hande: TODO: Calculate kl and entropy for loss
        # TODO: Convert code from torch to jax/flax
        # Stochastic Bert:  compute NLL, KL and entropy losses
        loss = None
        kl = None
        entropy = None
        if labels is not None:
            if n_samples > 1:
                labels = torch.repeat_interleave(labels, n_samples, dim=0)
            loss = D.Categorical(logits=logits).log_prob(labels).mean()
            kl, entropy = self.kl_and_entropy(self.config.kl_type, self.config.entropy_type)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


        if not return_dict:
            return (logits,) + outputs[2:]

        return FlaxStoSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

            #Hande: TODO: add kl and entropy loss
        )


        return loss

    grad_function = jax.value_and_grad(loss_function)
    loss, grad = grad_function(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean({"loss": loss, "learning_rate": learning_rate_function(state.step)}, axis_name="batch")
    return new_state, metrics, new_train_step_rng



def main():

    args = parse_args()


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # If passed along, set the training seed now.
    #if args.seed is not None:
    #    set_seed(args.seed)

    # RNG
    rng = jax.random.PRNGKey(args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Tokenizer
    config = StoBertConfig()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # Model
    #model = StoBertForSequenceClassification(config).from_pretrained(args.model_name_or_path, config=config, num_labels=3)
    config.num_labels = args.num_labels
    model = FlaxStoBertForSequenceClassification(config).from_pretrained(
        args.model_name_or_path, config=config
    )
   
    #prep data
    #prepare the train_loader, test_loader, eval_loader    
    # Data loaders
    config.dataset = args.dataset
    train_dataset, dev_dataset, test_dataset = get_nli_datasets(config, tokenizer, args.data_path)


    print('train_dataset:', len(train_dataset))
    print('dev_dataset:', len(dev_dataset))
    print('test_dataset:', len(test_dataset))



    #set up training parameters
    total_batch_size = args.train_batch_size * jax.local_device_count()
    print("The overall batch size (both for training and eval) is", total_batch_size)

    num_train_steps = len(train_dataset) // total_batch_size * arg.num_train_epochs

    learning_rate_function = optax.linear_schedule(init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps)

    #create the parallel_train_step function
    parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

    #TRAINING LOOP:
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=gradient_transformation,
        logits_function=eval_function,
        loss_function=loss_function,
    )


    for i, epoch in enumerate(tqdm(range(1, num_train_epochs + 1), desc=f"Epoch ...", position=0, leave=True)):

        rng, data_rng = jax.random.split(rng)
    
    
        # Train
        with tqdm(total=len(train_dataset) // total_batch_size, desc="Training...", leave=False) as progress_bar_train:
            for batch in train_data_loader(data_rng, train_dataset, total_batch_size):
                state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
                progress_bar_train.update(1)

        # Evaluate
        #with tqdm(total=len(eval_dataset) // total_batch_size, desc="Evaluating...", leave=False) as progress_bar_eval:
        #    for batch in eval_data_loader(dev_dataset, total_batch_size):
        #        labels = batch.pop("labels")
        #        predictions = parallel_eval_step(state, batch)
        #        metric.add_batch(predictions=chain(*predictions), references=chain(*labels))
        #        progress_bar_eval.update(1)

        #eval_metric = metric.compute()

        #loss = round(flax.jax_utils.unreplicate(train_metrics)['loss'].item(), 3)
        #eval_score = round(list(eval_metric.values())[0], 3)
        #metric_name = list(eval_metric.keys())[0]

        #print(f"{i+1}/{num_train_epochs} | Train loss: {loss} | Eval {metric_name}: {eval_score}")      



    import sys
    sys.exit(1)

    
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
