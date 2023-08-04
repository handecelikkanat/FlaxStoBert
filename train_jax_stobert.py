import argparse
import logging
import math
import os
import sys
import random

from pathlib import Path
import datasets
import torch
from tqdm.auto import tqdm
import numpy as np
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

import jax.numpy as jnp

from flax import traverse_util
from flax.training.common_utils import get_metrics, onehot
from flax.training import train_state

from typing import Callable

#from models.modeling_bert import BertForSequenceClassification
from models.modeling_flax_stobert import (
        FlaxStoBertForSequenceClassification, 
        FlaxStoSequenceClassifierOutput,
)
#from models_nli.modeling_outputs import StoSequenceClassifierOutput
from models.config import StoBertConfig
from data_nli import get_nli_datasets, train_data_loader, eval_data_loader

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
            "mnli-m-tiny-chaosnli",  # 96 train examples
            "mnli-m-16-chaosnli",  # 16 train examples
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
    parser.add_argument(
        "--gpu_devices",
        type=int,
        default=None,
        nargs='*',
        help="gpu devices to be used. ex: 0, 1",
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


class TrainState(train_state.TrainState):
    logits_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def adamw(weight_decay, learning_rate_function):
    return optax.adamw(learning_rate=learning_rate_function, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay, mask=decay_mask_fn)


def loss_function(logits, labels, num_labels, categorical_rng):
    xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))

    #also calculate kl and entropy for the loss
    # Stochastic Bert:  compute NLL, KL and entropy losses
    #loss = None
    #kl = None
    #entropy = None
    #if labels is not None:
    #    if n_samples > 1:
    #        labels = torch.repeat_interleave(labels, n_samples, dim=0)
    #    loss = D.Categorical(logits=logits).log_prob(labels).mean()
    #    kl, entropy = self.kl_and_entropy(self.config.kl_type, self.config.entropy_type)

    return jnp.mean(xentropy)


def eval_function(logits):
    return logits[..., 0] if is_regression else logits.argmax(-1)


# Create the parallel_train_step function
def train_step(state, batch, num_label, learning_rate_function, train_step_rng):
    targets = batch.pop("labels")

    dropout_rng, low_rank_rng, categorical_rng, new_train_step_rng = jax.random.split(train_step_rng, 4)

    def calculate_loss(params):
    # TODO: out apply function returns FlaxStoSequenceClassifierOutput
        output = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, low_rank_rng=low_rank_rng, train=True)
        logits = output.logits
        loss = state.loss_function(logits, targets, num_labels, categorical_rng)
        return loss

    grad_function = jax.value_and_grad(calculate_loss)
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

    # Initialize JAX
    print(args.gpu_devices)
    jax.distributed.initialize(local_device_ids=args.gpu_devices)  # On GPU, see above for the necessary arguments.
    print('jax device count:', jax.device_count())  # total number of accelerator devices in the cluster
    print('jax local device count: ', jax.local_device_count())  # number of accelerator devices attached to this host

    sys.exit(1)

    # RNG
    rng = jax.random.PRNGKey(args.seed)
    train_step_rng, data_rng = jax.random.split(rng, 2)
    train_step_rngs = jax.random.split(train_step_rng, jax.local_device_count())

    # Tokenizer
    config = StoBertConfig()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # Model
    config.num_labels = args.num_labels
    model = FlaxStoBertForSequenceClassification(config).from_pretrained(
        args.model_name_or_path, config=config, seed=0
    )
   
    # Datasets
    config.dataset = args.dataset
    train_dataset, dev_dataset, test_dataset = get_nli_datasets(config, tokenizer, args.data_path)

    # Training parameters
    total_batch_size = args.train_batch_size * jax.local_device_count()
    print("The overall batch size (both for training and eval) is", total_batch_size)

    num_train_steps = len(train_dataset) // total_batch_size * args.num_train_epochs

    learning_rate_function = optax.linear_schedule(init_value=args.learning_rate, end_value=0, transition_steps=num_train_steps)

    # Paralellize train function
    parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

    # Training Loop
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=adamw(weight_decay=0.01, learning_rate_function=learning_rate_function),
        logits_function=eval_function,
        loss_function=loss_function,
    )

    state = flax.jax_utils.replicate(state)
    num_labels = flax.jax_utils.replicate(args.num_labels)

    myindex = 1
    for i, epoch in enumerate(tqdm(range(1, args.num_train_epochs + 1), desc=f"Epoch ...", position=0, leave=True)):

        data_rng, data_rng_to_be_used = jax.random.split(data_rng)

        # Train
        with tqdm(total=len(train_dataset) // total_batch_size, desc="Training...", leave=False) as progress_bar_train:
            for batch in train_data_loader(data_rng_to_be_used, train_dataset, total_batch_size):
                state, train_metrics, train_step_rngs = parallel_train_step(state, batch, num_labels, train_step_rngs)
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

'''    
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
'''


if __name__ == "__main__":
    main()
