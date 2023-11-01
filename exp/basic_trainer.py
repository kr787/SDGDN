import torch
import math
import os
import time
import copy
import numpy as np

from exp.logger import get_logger
from exp.metrics import all_metrics
from exp.data_factory import NYCO, NYCD, PEMS04, PEMS07, PEMS08, BJD, BJO
from exp.model_factory import AGCRN, ASTGNN, DCRNN, DGCRN, DMSTGCN, GCDE, GWN, HA, MTGNN, SCINET, SDGDN
from exp.costs import get_model_parameters, get_memory_usage, write_cost


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.forward = self.forward_strategy(args)
        self.split = self.split_strategy(args)
        # if val_loader != None:
        #     self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        # self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        # self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        if args.model in [MTGNN, DGCRN] and args.cl:
            self.updata_cl_level = self.get_cl_level_by_iter()
        # if not args.debug:
        #     self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    @staticmethod
    def split_strategy(args):
        model = args.model

        if SDGDN in model or model in [AGCRN, ASTGNN, DCRNN, HA, SCINET]:
            def f(batch_data):
                data, target = batch_data
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                return data, target, label
        elif model == DGCRN:
            def f(batch_data):
                data, target = batch_data
                data = data.transpose(1, 3)
                label = target[..., :args.output_dim]
                target = target.transpose(1, 3)
                return data, target, label
        elif model == GCDE:
            def f(batch_data):
                batch_data = tuple(b.to(args.device, dtype=torch.float) for b in batch_data)
                *train_coeffs, target = batch_data
                label = target[..., :args.output_dim]
                return train_coeffs, target, label
        elif model in [GWN, MTGNN, DMSTGCN]:
            def f(batch_data):
                data, target = batch_data
                data = data.transpose(1, 3)
                label = target[..., :args.output_dim]
                return data, target, label
        else:
            raise ValueError
        return f

    @staticmethod
    def forward_strategy(args, show=False):
        model_type = args.model
        if SDGDN in model_type or model_type in [AGCRN, DCRNN]:
            def f(model, data, target, **kwargs):
                output = model(data, target)
                return output
        elif ASTGNN == model_type:
            def f(model, data, target=None, fine_tune=True):
                encoder_inputs = data.transpose(1, 2)
                if fine_tune:  # val, test, or later training process
                    decoder_inputs = encoder_inputs[:, :, -1:, :]
                    encoder_output = model.encode(encoder_inputs)
                    first_decoder_input = decoder_inputs[:, :, :1, :]
                    decoder_input_list = [first_decoder_input]
                    for _ in range(encoder_inputs.shape[2]):
                        decoder_inputs = torch.cat(decoder_input_list, dim=2)
                        predict_output = model.decode(decoder_inputs, encoder_output)
                        decoder_input_list = [first_decoder_input, predict_output]
                else:  # earlier training process
                    decoder_inputs = torch.cat((encoder_inputs[:, :, -1:, :], target.transpose(1, 2)[:, :, :-1, :]),
                                               dim=2)
                    predict_output = model(encoder_inputs, decoder_inputs)

                predict_output = predict_output.transpose(1, 2)
                return predict_output
        elif DGCRN == model_type:  # deprecated
            def f(model, data, target=None, batches_seen=None, task_level=args.output_window):
                if batches_seen:
                    output = model(data, ycl=target, batches_seen=batches_seen, task_level=task_level)
                else:
                    output = model(data, ycl=target, task_level=task_level)
                return output
        elif DMSTGCN == model_type:
            def f(model, data, **kwargs):
                inputs = data[:, :2, :, :]
                ind = data[:, 2, 0, 0].long()
                output = model(inputs, ind)
                return output
        elif GCDE == model_type:
            times = torch.linspace(0, 11, 12).to(args.device)

            def f(model, data, **kwargs):
                output = model(times, data)
                return output
        elif model_type in [GWN, HA]:
            def f(model, data, **kwargs):
                output = model(data)
                return output
        elif MTGNN == model_type:  # deprecated
            def f(model, data, id=None, **kwargs):
                if id is not None:
                    output = model(input=data, idx=id)
                else:
                    output = model(input=data)
                return output
        elif SCINET == model_type:
            def f(model, data, **kwargs):
                data = data.squeeze(-1)
                output = model(data)
                output = output.unsqueeze(-1)
                return output
        else:
            raise ValueError

        return f

    def get_cl_level_by_epoch(self, epoch):
        return self.args.output_window if (not hasattr(self.args, 'cl_step')) or self.args.cl_step == 0 else min(
            epoch // self.args.cl_step + 1, self.args.output_window)

    def get_cl_level_by_iter(self):  # deprecated, for MTGNN and DGCRN
        iter_ = 1
        task_level = 1

        def f():
            nonlocal iter_, task_level
            if iter_ % self.args.step_size1 == 0 and task_level < self.args.output_window:
                task_level += 1
            iter_ += 1
            return iter_, task_level

        return f

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_dataloader):
                data, target, label = self.split(batch_data)
                output = self.forward(model=self.model, data=data, target=target)
                output = output[:, :self.args.output_window, :, :]
                if self.args.real_value_test:
                    label = self.scaler.inverse_transform(label)
                if self.args.real_value_output_test:
                    output = self.scaler.inverse_transform(output)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)
        # self.logger.info('**Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def backward(self, output, label, cl_level):
        if self.args.real_value_output_train:
            output = self.scaler.inverse_transform(output)
        if self.args.real_value_train:
            label = self.scaler.inverse_transform(label)
        loss = self.loss(output[:, :cl_level, :, :], label[:, :cl_level, :, :])
        loss.backward()
        return loss.item()

    def train_epoch(self, epoch, fine_tune=True):
        self.model.train()
        total_loss = 0
        total_time = 0
        cl_level = self.get_cl_level_by_epoch(epoch)
        for batch_idx, batch_data in enumerate(self.train_loader):
            data, target, label = self.split(batch_data)
            self.optimizer.zero_grad()

            # deprecated
            # if self.args.model == MTGNN:
            #     if batch_idx % self.args.step_size2 == 0:
            #         perm = np.random.permutation(range(self.args.num_nodes))
            #     num_sub = int(self.args.num_nodes/self.args.num_split)
            #     for j in range(self.args.num_split):
            #         if j != self.args.num_split-1:
            #             id = perm[j * num_sub:(j + 1) * num_sub]
            #         else:
            #             id = perm[j * num_sub:]
            #         id = torch.tensor(id).to(self.args.device)
            #         begin = time.time()
            #         output = self.forward(model=self.model, data=data[:, :, id, :], id=id)
            #         _, cl_level = self.updata_cl_level()
            #         total_loss += self.backward(output, label, cl_level)
            # elif self.args.model == DGCRN:
            #     iter_, cl_level = self.updata_cl_level()
            #     import copy
            #     ycl = copy.deepcopy(target)
            #     ycl[..., 0] = self.scaler.transform(ycl[..., 0])
            #     begin = time.time()
            #     output = self.forward(model=self.model, data=data, target=ycl, batches_seen=iter_, task_level=cl_level)
            #     total_loss += self.backward(output, label, cl_level)
            # else:

            begin = time.time()
            output = self.forward(model=self.model, data=data, target=target, fine_tune=fine_tune)
            total_loss += self.backward(output, label, cl_level)

            total_time += time.time() - begin

            # add max grad clipping
            if hasattr(self.args, 'grad_norm') and self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # log information
            # if batch_idx % self.args.log_step == 0:
            #     self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
            #         epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        # self.logger.info('**Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        # learning rate decay
        if self.args.model == SCINET:
            import SCINet.utils.tools
            SCINet.utils.tools.adjust_learning_rate(self.optimizer, epoch, self.args)
            if epoch % self.args.exponential_decay_step == 0:
                self.lr_scheduler.step()
        elif hasattr(self.args, 'lr_decay') and self.args.lr_decay:
            self.lr_scheduler.step()

        if self.args.mode == 'train_cost':
            write_cost(self.args.model, self.args.dataset, 'train_cost_time', total_time)
            total_num = get_model_parameters(self.model, show=False)
            write_cost(self.args.model, self.args.dataset, 'parameters', total_num)
            allocated_memory, cached_memory = get_memory_usage(self.args.device)
            write_cost(self.args.model, self.args.dataset, 'train_allocated_memory', allocated_memory)
            write_cost(self.args.model, self.args.dataset, 'train_cached_memory', cached_memory)

        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float("inf")
        not_improved_count = 0
        need_fine_tune = hasattr(self.args, 'fine_tune_epochs')
        if need_fine_tune:  # only for ASTGNN
            self.args.epochs += self.args.fine_tune_epochs

        start_time = time.time()
        self.logger.info("Epoch\tTrain Loss\tVal Loss\tlr\tTest")

        for epoch in range(1, self.args.epochs + 1):
            fine_tune_epoch = need_fine_tune and epoch + self.args.fine_tune_epochs > self.args.epochs
            if fine_tune_epoch:
                train_epoch_loss = self.train_epoch(epoch, fine_tune=True)
            else:
                train_epoch_loss = self.train_epoch(epoch, fine_tune=False)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            best_state = False
            if (self.get_cl_level_by_epoch(epoch) < self.args.output_window) or (
                    need_fine_tune and not fine_tune_epoch):
                val_epoch_loss = float("inf")  # skip val to save time
            else:
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
                # early stop
                if hasattr(self.args, 'early_stop') and self.args.early_stop:
                    if not_improved_count == self.args.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve since epoch {}. "
                                         "Training stops.".format(epoch - self.args.early_stop_patience))
                        break

            log = "{}\t{:.6f}\t{:.6f}\t{:.6f}".format(epoch, train_epoch_loss, val_epoch_loss,
                                                      self.optimizer.param_groups[0]['lr'])
            # save the best state
            if best_state:
                log += " *Current best model! "
                # log += " ".join(self.test(self.model, self.args, self.test_loader, self.scaler))
                best_model = copy.deepcopy(self.model.state_dict())

            self.logger.info(log)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        # if not self.args.debug:
        #     torch.save(best_model, self.best_path)
        #     self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def test(model, args, data_loader, scaler, logger=None, path=None, week_idx=None):
        model.eval()
        y_pred = []
        y_true = []
        forward = Trainer.forward_strategy(args)
        split = Trainer.split_strategy(args)
        total_time = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                data, target, label = split(batch_data)
                if week_idx is not None and not batch_idx in week_idx:
                    continue
                begin = time.time()
                output = forward(model=model, data=data, target=target)
                total_time += time.time() - begin
                y_true.append(label)
                y_pred.append(output)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        if args.real_value_output_test:
            y_pred = scaler.inverse_transform(y_pred)
        if args.real_value_test:
            y_true = scaler.inverse_transform(y_true)
        logs = ["test result", "MAE\tRMSE\tMAPE\tHorizon"]

        y_pred = y_pred[:, :y_true.shape[1], :, :]
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = all_metrics(y_pred[:, t, :, 0], y_true[:, t, :, 0], args.mae_thresh,
                                                args.mape_thresh)
            logs.append("{:.2f}\t{:.2f}\t{:.2f}\t{:02d}".format(mae, rmse, mape * 100, t + 1))
        mae, rmse, mape, _, _ = all_metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logs.append("{:.2f}\t{:.2f}\t{:.2f}".format(mae, rmse, mape * 100))
        if logger:
            logger.info("\n".join(logs))

        if 'cost' in args.mode:
            write_cost(args.model, args.dataset, 'test_cost_time', total_time)
            allocated_memory, cached_memory = get_memory_usage(args.device)
            write_cost(args.model, args.dataset, 'test_allocated_memory', allocated_memory)
            write_cost(args.model, args.dataset, 'test_cached_memory', cached_memory)

        return logs
