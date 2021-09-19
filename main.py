import argparse
import glob
import logging
import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,RandomSampler,SequentialSampler,random_split
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn as nn
from utils import Record, get_metrics


from sklearn.metrics import *

import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers.trainer_utils import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter





logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_seq_length",default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=800, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=800, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--valset_ratio", type=float, default=0.2)



    parser.add_argument("--node_dim", type=int, default=128)
    parser.add_argument("--method",type=str, default="gat")
    parser.add_argument("--graph_order", type=int, default=1)

    parser.add_argument("--graph_type",type=str,default="tree")

    parser.add_argument("--is_parallel", action="store_true")
    # parser.add_argument("--pool_method", type=str, default=None, help="pooling from GNN ouput to classify input")
    # args = parser.parse_args(["--data_path=raw_data/test.csv",
    #                           "--output_dir=output/test",
    #                            "--do_eval","--do_train",
    #                           "--num_train_epochs=1.0", "--evaluate_during_training",
    #                           "--model_name_or_path=bert-base-uncased",
    #                           "--do_lower_case", "--eval_all_checkpoints",
    #                           "--logging_steps=1","--per_gpu_train_batch_size=1","--per_gpu_eval_batch_size=1",
    #                           "--method=nobert+gat+rnn","--is_parallel"])
    # args = parser.parse_args(["--data_path=pro_data/tweet/top1_tweet_0_1_ac.csv",
    #                           "--output_dir=output/test",
    #                           "--do_eval",
    #                           "--num_train_epochs=1.0", "--evaluate_during_training",
    #                           "--model_name_or_path=princeton-nlp/sup-simcse-bert-base-uncased",
    #                           "--do_lower_case", "--eval_all_checkpoints"])

    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset,  model, tokenizer, processor):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training ****self.classifier(clf_input)*")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    # TODO important
    set_seed(args)  # Added here for reproductibility

    # TODO best_loss
    best_loss = 100000
    best_steps = -1
    best_record = Record()
    best_acc = 0
    # TODO
    #  definite loss
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], miniters=100)
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        # count= 1
        for step, batch in enumerate(epoch_iterator):
            # logger.info("count: %s"%str(count))
            # count+=1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # TODO begin
            #  construct input data
            batch = tuple(t.to(args.device) for t in batch)

            if processor.processor_type == "vanilla":
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],"labels": batch[3]}
            if "graph" in processor.processor_type:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],"labels": batch[3],'dependency_matrix':batch[4]}

            # TODO
            #  calculate loss
            #  logits = model(**inputs)
            #  loss = loss_function(logits,inputs['labels'])

            outputs = model(**inputs)
            loss = loss_function(outputs, inputs['labels'])

            # TODO end
            # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_dataset, model, tokenizer,processor)
                        for key in ["accuracy", "macro_f1"]:
                            tb_writer.add_scalar("eval_{}".format(key), results[key], global_step)
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    # TODO
                    #  log & save steps equal

                    eval_loss = results["loss"]
                    eval_acc = results["accuracy"]
                    eval_macro_f1 = results["macro_f1"]
                    # if eval_loss < best_loss:
                    # if eval_acc > best_acc:
                    if eval_acc > best_record.accuracy:
                        logger.info("update best record")
                        best_acc = eval_acc
                        best_steps = global_step
                        best_record.update(**results)
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training

                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    logger.info("----------------------------------------")
    logger.info("stop training process")
    logger.info("best steps   : {}".format(best_steps))
    # logger.info("best loss   : {}".format(best_loss))

    logger.info("best accuracy: {:.4f}".format(best_record.accuracy))
    logger.info("best macro_F1: {:.4f}".format(best_record.macro_f1))
    logger.info("  - {:<8}: {:<9} | {:<9} | {:<9}".format("category", "precision", "recall", "f1_score"))
    for idx in range(processor.num_labels):
        logger.info("  - {:<8}: {:<9.4f} | {:<9.4f} | {:<9.4f}".format(processor.label_list[idx], best_record.precision[idx], best_record.recall[idx], best_record.f1_score[idx]))


    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, processor,prefix=""):
    # eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_output_dir in eval_outputs_dirs:
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # TODO begin
        #  definite loss
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)

        for batch in tqdm(eval_dataloader, desc="Evaluating", miniters=100):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                # TODO
                #  construct input data
                if processor.processor_type == "vanilla":
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                              "labels": batch[3]}
                if "graph" in processor.processor_type:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                              "labels": batch[3], 'dependency_matrix': batch[4]}
                outputs = model(**inputs)
                logits = outputs
                probs = nn.functional.softmax(logits)
                tmp_eval_loss = loss_function(outputs, inputs['labels'])

                # tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = probs.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        # TODO
        #  metrics(y_true, y_pred)
        result = get_metrics(out_label_ids,preds)
        result["loss"] = eval_loss
        results.update(result)
        # TODO
        #  end

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in ["accuracy", "macro_f1"]:
                writer.write("%s = %s\n" % (key, str(result[key])))
            logger.info("test accuracy:{:.4f}".format(result["accuracy"]))
            logger.info("test macro_F1:{:.4f}".format(result["macro_f1"]))
            logger.info(
                "  - {:<8}: {:<9} | {:<9} | {:<9}".format("category", "test precision", "test recall", "test f1_score"))
            for idx in range(processor.num_labels):
                logger.info(
                    "  - {:<8}: {:<9.4f} | {:<9.4f} | {:<9.4f}".format(processor.label_list[idx], result["precision"][idx],
                                                                       result["recall"][idx], result["f1_score"][idx]))

    return results


def main(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # TODO
    #  prepare data
    data_path = args.data_path
    logger.info("loading training dataset_d from {}...".format(data_path))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )

    if args.method in ["gat", "gat+fc","gat+cnn", "gat+rnn", "nobert+gat+fc","nobert+gat+cnn","nobert+gat+rnn"]:
        from graph_process import DataProcessor
        processor = DataProcessor(tokenizer, order=args.graph_order, type=args.graph_type)
    if args.method in ["cnn", "fc","rnn", "bert_sentence_fc"]:
        from process import DataProcessor
        processor = DataProcessor(tokenizer)


    dataset_total, examples_total = processor.get_examples(data_path)

    testset_len = int(len(dataset_total) * args.valset_ratio)
    train_val_dataset, test_dataset_d = random_split(dataset_total, (len(dataset_total) - testset_len, testset_len))

    valset_len = int(len(train_val_dataset) * args.valset_ratio)

    train_dataset_d, val_dataset_d = random_split(train_val_dataset,
                                                      (len(train_val_dataset) - valset_len, valset_len))


    logger.info("number of validation examples:{}".format(len(test_dataset_d)))

    # load model and tokenizer?
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir,
    )
    args.model_type = config.model_type

    if args.method in ["gat"]:
        from model_gat import TCModel
        options = {"node_dim":args.node_dim}
    if args.method in ["gat+fc"]:
        if args.is_parallel:
            from parallel.bert_gat_fc import TCModel
        else:
            from bert_gat_fc import TCModel
        options = {}
    if args.method in ["gat+cnn"]:
        if args.is_parallel:
            from parallel.bert_gat_cnn import  TCModel
        else:
            from bert_gat_cnn import TCModel
        options = {}
    if args.method in ["gat+rnn"]:
        if args.is_parallel:
            from parallel.bert_gat_rnn import TCModel
        else:
            from bert_gat_rnn import TCModel
        options = {}

    if args.method in ["nobert+gat+fc"]:
        from nobert.gat_fc import TCModel
        options = {}

    if args.method in ["nobert+gat+cnn"]:
        from nobert.gat_cnn import TCModel
        options = {}

    if args.method in ["nobert+gat+rnn"]:
        from nobert.gat_rnn import TCModel
        options = {}

    if args.method == "fc":
        from bert_fc import TCModel
        options = {}
    if args.method == "cnn":
        from bert_cnn import TCModel
        options = {}
    if args.method == "bert_sentence_fc":
        from bert_sentence_fc import TCModel
        options = {}
    if args.method == "rnn":
        from bert_rnn import TCModel
        options = {}



    model = TCModel.from_pretrained(args.model_name_or_path, **options)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset_d, val_dataset_d, model, tokenizer, processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = TCModel.from_pretrained(checkpoint, **options)
            model.to(args.device)
            result = evaluate(args, test_dataset_d, model, tokenizer, processor, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)


if __name__ == '__main__':
    args = get_args()
    main(args)